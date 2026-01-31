"""
Основной класс цезиевого магнитометра.

Высокоуровневый интерфейс для работы с цезиевым магнитометром,
объединяющий драйвер STM32, AI-обработку через Hailo и
интеграцию с BlueOS.

Принцип работы:
    Цезиевый магнитометр использует оптическую накачку атомов Cs-133
    для измерения магнитного поля. Частота ларморовской прецессии
    пропорциональна напряжённости магнитного поля:

    f = γ × B

    где γ = 3.498 Гц/нТл — гиромагнитное отношение цезия-133,
    B — напряжённость магнитного поля.

Author: НПО Лаборатория К
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .config import (
    CalibrationData,
    FilterType,
    MagnetometerConfig,
    MeasurementMode,
)
from .stm32_driver import MagnetometerReading, STM32Driver, STM32Status


logger = logging.getLogger(__name__)


@dataclass
class ProcessedReading:
    """Обработанные данные измерения магнитометра.

    Attributes:
        timestamp: Временная метка (datetime)
        raw_value: Сырое значение поля в нТл
        filtered_value: Отфильтрованное значение в нТл
        calibrated_value: Откалиброванное значение в нТл
        gradient: Градиент поля в нТл/м (если доступен)
        latitude: Широта (если доступна от BlueOS)
        longitude: Долгота (если доступна от BlueOS)
        depth: Глубина в метрах (если доступна)
        quality: Качество измерения (0-100%)
        anomaly_detected: Флаг обнаружения аномалии (AI)
        anomaly_confidence: Уверенность обнаружения аномалии (0-1)
    """

    timestamp: datetime
    raw_value: float
    filtered_value: float
    calibrated_value: float
    gradient: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    quality: int = 100
    anomaly_detected: bool = False
    anomaly_confidence: float = 0.0


@dataclass
class MagnetometerStatistics:
    """Статистика измерений магнитометра.

    Attributes:
        samples_count: Общее количество измерений
        samples_per_second: Фактическая частота измерений
        mean_value: Среднее значение поля в нТл
        std_deviation: Стандартное отклонение в нТл
        min_value: Минимальное значение в нТл
        max_value: Максимальное значение в нТл
        anomalies_detected: Количество обнаруженных аномалий
        uptime_seconds: Время работы в секундах
    """

    samples_count: int = 0
    samples_per_second: float = 0.0
    mean_value: float = 0.0
    std_deviation: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    anomalies_detected: int = 0
    uptime_seconds: float = 0.0


class KalmanFilter:
    """Фильтр Калмана для обработки данных магнитометра.

    Одномерный фильтр Калмана для сглаживания показаний
    магнитометра и подавления шума.

    Attributes:
        process_variance: Дисперсия процесса
        measurement_variance: Дисперсия измерений
    """

    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.1,
    ):
        """Инициализация фильтра Калмана.

        Args:
            process_variance: Дисперсия процесса (шум модели)
            measurement_variance: Дисперсия измерений (шум датчика)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # Состояние фильтра
        self._estimate: Optional[float] = None
        self._error_estimate: float = 1.0

    def update(self, measurement: float) -> float:
        """Обновление фильтра новым измерением.

        Args:
            measurement: Новое измерение

        Returns:
            float: Отфильтрованное значение
        """
        if self._estimate is None:
            # Инициализация первым измерением
            self._estimate = measurement
            return measurement

        # Предсказание
        prediction = self._estimate
        prediction_error = self._error_estimate + self.process_variance

        # Коррекция
        kalman_gain = prediction_error / (
            prediction_error + self.measurement_variance
        )

        self._estimate = prediction + kalman_gain * (measurement - prediction)
        self._error_estimate = (1 - kalman_gain) * prediction_error

        return self._estimate

    def reset(self) -> None:
        """Сброс состояния фильтра."""
        self._estimate = None
        self._error_estimate = 1.0


class CesiumMagnetometer:
    """Основной класс цезиевого магнитометра.

    Предоставляет высокоуровневый интерфейс для работы с
    цезиевым магнитометром, включая:
    - Подключение и настройку устройства
    - Получение и обработку измерений
    - Применение фильтров и калибровки
    - AI-обработку для обнаружения аномалий
    - Интеграцию с BlueOS

    Attributes:
        config: Конфигурация магнитометра
        is_connected: Флаг состояния подключения
        is_measuring: Флаг режима измерения
        statistics: Текущая статистика измерений

    Example:
        >>> mag = CesiumMagnetometer()
        >>> mag.connect()
        >>> mag.start_measurement(callback=lambda r: print(r.calibrated_value))
        >>> # ... измерения выполняются в фоне ...
        >>> mag.stop_measurement()
        >>> mag.disconnect()
    """

    def __init__(self, config: Optional[MagnetometerConfig] = None):
        """Инициализация магнитометра.

        Args:
            config: Конфигурация магнитометра.
                    Если не указана, используются значения по умолчанию.
        """
        self.config = config or MagnetometerConfig()
        self.config.validate()

        # Компоненты
        self._driver: Optional[STM32Driver] = None
        self._hailo_processor: Optional["HailoProcessor"] = None
        self._blueos_extension: Optional["BlueOSExtension"] = None

        # Фильтры
        self._kalman_filter = KalmanFilter()
        self._filter_buffer: List[float] = []

        # Состояние
        self.is_connected: bool = False
        self.is_measuring: bool = False
        self.statistics = MagnetometerStatistics()

        # Внутренние переменные
        self._data_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._user_callback: Optional[Callable[[ProcessedReading], None]] = None
        self._start_time: Optional[float] = None
        self._readings_buffer: List[float] = []

        # Настройка логирования
        logging.basicConfig(level=getattr(logging, self.config.log_level))

        logger.info(
            "CesiumMagnetometer инициализирован: sample_rate=%d, mode=%s",
            self.config.sample_rate,
            self.config.measurement_mode.name,
        )

    def connect(self) -> bool:
        """Подключение к магнитометру.

        Устанавливает соединение с STM32, инициализирует
        AI-процессор Hailo (если включён) и регистрирует
        расширение в BlueOS.

        Returns:
            bool: True если подключение успешно

        Raises:
            ConnectionError: Ошибка подключения
        """
        if self.is_connected:
            logger.warning("Магнитометр уже подключён")
            return True

        try:
            # Подключение к STM32
            self._driver = STM32Driver(self.config.serial)
            self._driver.connect()

            # Настройка частоты измерений
            self._driver.set_sample_rate(self.config.sample_rate)

            # Инициализация Hailo (если включён)
            if self.config.hailo.enabled:
                self._init_hailo()

            # Создание директории для данных
            Path(self.config.data_directory).mkdir(parents=True, exist_ok=True)

            self.is_connected = True
            self._start_time = time.time()

            logger.info("Магнитометр подключён успешно")
            return True

        except Exception as e:
            logger.error("Ошибка подключения магнитометра: %s", e)
            self.disconnect()
            raise ConnectionError(f"Ошибка подключения: {e}") from e

    def disconnect(self) -> None:
        """Отключение от магнитометра.

        Останавливает измерения, закрывает соединения
        и освобождает ресурсы.
        """
        if self.is_measuring:
            self.stop_measurement()

        if self._driver:
            self._driver.disconnect()
            self._driver = None

        if self._hailo_processor:
            self._hailo_processor.shutdown()
            self._hailo_processor = None

        self.is_connected = False
        logger.info("Магнитометр отключён")

    def get_single_reading(self) -> ProcessedReading:
        """Получение одиночного измерения.

        Выполняет одиночное измерение, применяет калибровку
        и фильтрацию.

        Returns:
            ProcessedReading: Обработанное измерение

        Raises:
            ConnectionError: Магнитометр не подключён
        """
        self._check_connection()

        raw_reading = self._driver.get_reading()
        return self._process_reading(raw_reading)

    def start_measurement(
        self,
        callback: Optional[Callable[[ProcessedReading], None]] = None,
    ) -> bool:
        """Запуск непрерывного измерения.

        Args:
            callback: Функция обратного вызова для каждого измерения.
                      Если не указана, данные накапливаются во внутреннем
                      буфере и доступны через get_buffered_readings().

        Returns:
            bool: True если измерение запущено

        Example:
            >>> def on_reading(reading):
            ...     if reading.anomaly_detected:
            ...         print(f"Anomaly at {reading.latitude}, {reading.longitude}")
            >>> mag.start_measurement(callback=on_reading)
        """
        self._check_connection()

        if self.is_measuring:
            logger.warning("Измерение уже запущено")
            return False

        self._user_callback = callback
        self._stop_event.clear()

        # Запуск потока обработки
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
        )
        self._processing_thread.start()

        # Запуск измерения на STM32
        success = self._driver.start_continuous(
            callback=self._on_raw_reading,
        )

        if success:
            self.is_measuring = True
            logger.info("Непрерывное измерение запущено")

        return success

    def stop_measurement(self) -> bool:
        """Остановка непрерывного измерения.

        Returns:
            bool: True если измерение остановлено
        """
        if not self.is_measuring:
            return True

        self._stop_event.set()

        if self._driver:
            self._driver.stop_continuous()

        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
            self._processing_thread = None

        self.is_measuring = False
        logger.info("Непрерывное измерение остановлено")
        return True

    def calibrate(self) -> bool:
        """Запуск калибровки магнитометра.

        Выполняет автоматическую калибровку устройства.
        Во время калибровки магнитометр должен находиться
        в стабильном магнитном поле без помех.

        Returns:
            bool: True если калибровка успешна
        """
        self._check_connection()

        logger.info("Запуск калибровки...")
        success = self._driver.calibrate()

        if success:
            # Обновление данных калибровки
            self.config.calibration.last_calibration = datetime.now().isoformat()
            logger.info("Калибровка завершена успешно")
        else:
            logger.error("Ошибка калибровки")

        return success

    def get_status(self) -> dict:
        """Получение текущего статуса магнитометра.

        Returns:
            dict: Словарь со статусной информацией

        Example:
            >>> status = mag.get_status()
            >>> print(f"Connected: {status['connected']}")
            >>> print(f"Firmware: {status['firmware_version']}")
        """
        status = {
            "connected": self.is_connected,
            "measuring": self.is_measuring,
            "firmware_version": (
                self._driver.firmware_version if self._driver else "N/A"
            ),
            "hailo_enabled": self.config.hailo.enabled,
            "sample_rate": self.config.sample_rate,
            "measurement_mode": self.config.measurement_mode.name,
            "filter_type": self.config.filter_type.name,
        }

        if self.is_connected and self._driver:
            try:
                device_status = self._driver.get_status()
                status["device_status"] = device_status.name
            except Exception:
                status["device_status"] = "UNKNOWN"

        return status

    def get_statistics(self) -> MagnetometerStatistics:
        """Получение статистики измерений.

        Returns:
            MagnetometerStatistics: Текущая статистика
        """
        if self._start_time:
            self.statistics.uptime_seconds = time.time() - self._start_time

        if self._readings_buffer:
            arr = np.array(self._readings_buffer[-1000:])  # Последние 1000
            self.statistics.mean_value = float(np.mean(arr))
            self.statistics.std_deviation = float(np.std(arr))
            self.statistics.min_value = float(np.min(arr))
            self.statistics.max_value = float(np.max(arr))

        return self.statistics

    def get_buffered_readings(
        self,
        max_count: Optional[int] = None,
    ) -> List[ProcessedReading]:
        """Получение накопленных измерений из буфера.

        Args:
            max_count: Максимальное количество измерений.
                       Если не указано, возвращаются все.

        Returns:
            List[ProcessedReading]: Список измерений
        """
        readings = []
        count = 0

        while not self._data_queue.empty():
            if max_count and count >= max_count:
                break

            try:
                reading = self._data_queue.get_nowait()
                readings.append(reading)
                count += 1
            except queue.Empty:
                break

        return readings

    def set_filter(
        self,
        filter_type: FilterType,
        cutoff_hz: Optional[float] = None,
    ) -> None:
        """Установка типа фильтра.

        Args:
            filter_type: Тип фильтра
            cutoff_hz: Частота среза в Гц (для LOWPASS, BANDPASS)
        """
        self.config.filter_type = filter_type

        if cutoff_hz is not None:
            self.config.filter_cutoff = cutoff_hz

        # Сброс состояния фильтра
        self._kalman_filter.reset()
        self._filter_buffer.clear()

        logger.info(
            "Установлен фильтр: %s, cutoff=%.1f Гц",
            filter_type.name,
            self.config.filter_cutoff,
        )

    def export_data(
        self,
        filepath: str,
        format: str = "csv",
    ) -> bool:
        """Экспорт накопленных данных в файл.

        Args:
            filepath: Путь к файлу
            format: Формат файла ('csv', 'json')

        Returns:
            bool: True если экспорт успешен
        """
        readings = self.get_buffered_readings()

        if not readings:
            logger.warning("Нет данных для экспорта")
            return False

        try:
            if format == "csv":
                self._export_csv(filepath, readings)
            elif format == "json":
                self._export_json(filepath, readings)
            else:
                raise ValueError(f"Неподдерживаемый формат: {format}")

            logger.info("Данные экспортированы в %s", filepath)
            return True

        except Exception as e:
            logger.error("Ошибка экспорта данных: %s", e)
            return False

    def _check_connection(self) -> None:
        """Проверка подключения магнитометра.

        Raises:
            ConnectionError: Магнитометр не подключён
        """
        if not self.is_connected:
            raise ConnectionError("Магнитометр не подключён")

    def _init_hailo(self) -> None:
        """Инициализация AI-процессора Hailo."""
        try:
            from .hailo_processor import HailoProcessor

            self._hailo_processor = HailoProcessor(self.config.hailo)
            self._hailo_processor.initialize()
            logger.info("Hailo AI-процессор инициализирован")
        except ImportError:
            logger.warning(
                "Модуль hailo_processor не найден, AI отключён"
            )
            self.config.hailo.enabled = False
        except Exception as e:
            logger.warning("Ошибка инициализации Hailo: %s", e)
            self.config.hailo.enabled = False

    def _on_raw_reading(self, reading: MagnetometerReading) -> None:
        """Callback для сырых данных от STM32.

        Args:
            reading: Сырое измерение
        """
        try:
            processed = self._process_reading(reading)

            # Добавление в очередь
            if not self._data_queue.full():
                self._data_queue.put(processed)

            # Обновление буфера для статистики
            self._readings_buffer.append(processed.calibrated_value)
            if len(self._readings_buffer) > 10000:
                self._readings_buffer = self._readings_buffer[-5000:]

            # Обновление статистики
            self.statistics.samples_count += 1

        except Exception as e:
            logger.error("Ошибка обработки измерения: %s", e)

    def _process_reading(
        self,
        raw_reading: MagnetometerReading,
    ) -> ProcessedReading:
        """Обработка сырого измерения.

        Args:
            raw_reading: Сырое измерение от STM32

        Returns:
            ProcessedReading: Обработанное измерение
        """
        raw_value = raw_reading.field_total

        # Применение калибровки
        calibrated = self._apply_calibration(raw_value, raw_reading.temperature)

        # Применение фильтра
        filtered = self._apply_filter(calibrated)

        # AI-обработка (обнаружение аномалий)
        anomaly_detected = False
        anomaly_confidence = 0.0

        if self._hailo_processor and self.config.filter_type == FilterType.AI_DENOISE:
            anomaly_detected, anomaly_confidence = (
                self._hailo_processor.detect_anomaly(filtered)
            )
            if anomaly_detected:
                self.statistics.anomalies_detected += 1

        return ProcessedReading(
            timestamp=datetime.now(),
            raw_value=raw_value,
            filtered_value=filtered,
            calibrated_value=calibrated,
            quality=raw_reading.signal_quality,
            anomaly_detected=anomaly_detected,
            anomaly_confidence=anomaly_confidence,
        )

    def _apply_calibration(
        self,
        value: float,
        temperature: float,
    ) -> float:
        """Применение калибровки к измерению.

        Args:
            value: Сырое значение в нТл
            temperature: Температура сенсора в °C

        Returns:
            float: Откалиброванное значение
        """
        cal = self.config.calibration

        # Температурная компенсация
        temp_offset = cal.temperature_coefficient * (temperature - 25.0)

        # Применение смещения (для полного поля используем только offset_x)
        calibrated = (value - cal.offset_x) * cal.scale_x - temp_offset

        return calibrated

    def _apply_filter(self, value: float) -> float:
        """Применение цифрового фильтра.

        Args:
            value: Входное значение

        Returns:
            float: Отфильтрованное значение
        """
        filter_type = self.config.filter_type

        if filter_type == FilterType.NONE:
            return value

        elif filter_type == FilterType.KALMAN:
            return self._kalman_filter.update(value)

        elif filter_type == FilterType.LOWPASS:
            return self._lowpass_filter(value)

        elif filter_type == FilterType.AI_DENOISE:
            if self._hailo_processor:
                return self._hailo_processor.denoise(value)
            return value

        return value

    def _lowpass_filter(self, value: float) -> float:
        """Простой фильтр низких частот (скользящее среднее).

        Args:
            value: Входное значение

        Returns:
            float: Отфильтрованное значение
        """
        self._filter_buffer.append(value)

        # Размер окна зависит от частоты среза
        window_size = max(
            1,
            int(self.config.sample_rate / self.config.filter_cutoff),
        )

        if len(self._filter_buffer) > window_size:
            self._filter_buffer = self._filter_buffer[-window_size:]

        return float(np.mean(self._filter_buffer))

    def _processing_loop(self) -> None:
        """Цикл обработки данных в отдельном потоке."""
        logger.debug("Поток обработки данных запущен")

        last_stats_time = time.time()

        while not self._stop_event.is_set():
            try:
                # Получение данных из очереди с таймаутом
                try:
                    reading = self._data_queue.get(timeout=0.1)

                    # Вызов пользовательского callback
                    if self._user_callback:
                        try:
                            self._user_callback(reading)
                        except Exception as e:
                            logger.error("Ошибка в callback: %s", e)

                except queue.Empty:
                    continue

                # Периодическое обновление статистики
                if time.time() - last_stats_time > 1.0:
                    self._update_statistics()
                    last_stats_time = time.time()

            except Exception as e:
                logger.error("Ошибка в цикле обработки: %s", e)

        logger.debug("Поток обработки данных остановлен")

    def _update_statistics(self) -> None:
        """Обновление статистики измерений."""
        if self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self.statistics.samples_per_second = (
                    self.statistics.samples_count / elapsed
                )

    def _export_csv(
        self,
        filepath: str,
        readings: List[ProcessedReading],
    ) -> None:
        """Экспорт данных в CSV."""
        import csv

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "raw_value_nT",
                "filtered_value_nT",
                "calibrated_value_nT",
                "quality",
                "anomaly_detected",
                "anomaly_confidence",
            ])

            for r in readings:
                writer.writerow([
                    r.timestamp.isoformat(),
                    r.raw_value,
                    r.filtered_value,
                    r.calibrated_value,
                    r.quality,
                    r.anomaly_detected,
                    r.anomaly_confidence,
                ])

    def _export_json(
        self,
        filepath: str,
        readings: List[ProcessedReading],
    ) -> None:
        """Экспорт данных в JSON."""
        import json

        data = [
            {
                "timestamp": r.timestamp.isoformat(),
                "raw_value_nT": r.raw_value,
                "filtered_value_nT": r.filtered_value,
                "calibrated_value_nT": r.calibrated_value,
                "quality": r.quality,
                "anomaly_detected": r.anomaly_detected,
                "anomaly_confidence": r.anomaly_confidence,
            }
            for r in readings
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def __enter__(self) -> "CesiumMagnetometer":
        """Поддержка контекстного менеджера."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Закрытие при выходе из контекста."""
        self.disconnect()
