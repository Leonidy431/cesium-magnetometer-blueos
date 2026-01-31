"""
AI-процессор на базе Hailo 8L для магнитометра.

Модуль обеспечивает AI-обработку данных магнитометра с использованием
нейроускорителя Hailo 8L для:
- Шумоподавления сигнала
- Обнаружения магнитных аномалий
- Классификации источников аномалий

Hailo 8L — edge AI-ускоритель с производительностью до 13 TOPS
при энергопотреблении менее 2.5 Вт.

Author: НПО Лаборатория К
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

from .config import HailoConfig


logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Типы магнитных аномалий.

    Attributes:
        NONE: Аномалия не обнаружена
        FERROUS_OBJECT: Железосодержащий объект
        GEOLOGICAL: Геологическая аномалия
        INFRASTRUCTURE: Инфраструктура (трубопроводы, кабели)
        UNKNOWN: Неизвестная аномалия
    """

    NONE = auto()
    FERROUS_OBJECT = auto()
    GEOLOGICAL = auto()
    INFRASTRUCTURE = auto()
    UNKNOWN = auto()


@dataclass
class AnomalyDetection:
    """Результат обнаружения аномалии.

    Attributes:
        detected: Флаг обнаружения
        confidence: Уверенность (0-1)
        anomaly_type: Тип аномалии
        magnitude: Амплитуда аномалии в нТл
        description: Текстовое описание
    """

    detected: bool
    confidence: float
    anomaly_type: AnomalyType
    magnitude: float
    description: str


class HailoProcessor:
    """AI-процессор на базе Hailo 8L.

    Обеспечивает инференс нейронных сетей для обработки
    данных магнитометра в реальном времени.

    Attributes:
        config: Конфигурация Hailo
        is_initialized: Флаг инициализации
        model_loaded: Флаг загрузки модели

    Example:
        >>> processor = HailoProcessor(HailoConfig())
        >>> processor.initialize()
        >>> denoised = processor.denoise(value)
        >>> detected, confidence = processor.detect_anomaly(value)
    """

    # Размер буфера для анализа временного ряда
    BUFFER_SIZE = 256
    # Порог обнаружения аномалии
    ANOMALY_THRESHOLD = 0.7

    def __init__(self, config: Optional[HailoConfig] = None):
        """Инициализация AI-процессора.

        Args:
            config: Конфигурация Hailo.
                    Если не указана, используются значения по умолчанию.
        """
        self.config = config or HailoConfig()

        # Состояние
        self.is_initialized: bool = False
        self.model_loaded: bool = False

        # Hailo SDK объекты
        self._device = None
        self._hef = None
        self._network_group = None
        self._input_vstreams = None
        self._output_vstreams = None

        # Буферы данных
        self._data_buffer: Deque[float] = deque(maxlen=self.BUFFER_SIZE)
        self._inference_lock = threading.Lock()

        # Статистика для адаптивного порога
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1.0
        self._samples_for_baseline: List[float] = []

        logger.debug(
            "HailoProcessor создан: device_id=%d, model=%s",
            self.config.device_id,
            self.config.model_path,
        )

    def initialize(self) -> bool:
        """Инициализация Hailo устройства и загрузка модели.

        Returns:
            bool: True если инициализация успешна

        Raises:
            RuntimeError: Ошибка инициализации Hailo
        """
        if self.is_initialized:
            logger.warning("Hailo уже инициализирован")
            return True

        try:
            # Попытка импорта Hailo SDK
            try:
                from hailo_platform import (
                    HEF,
                    ConfigureParams,
                    FormatType,
                    HailoStreamInterface,
                    InputVStreamParams,
                    OutputVStreamParams,
                    VDevice,
                )

                self._hailo_available = True
            except ImportError:
                logger.warning(
                    "Hailo SDK не установлен, используется эмуляция"
                )
                self._hailo_available = False
                self.is_initialized = True
                return True

            # Создание виртуального устройства
            params = VDevice.create_params()
            params.device_ids = [self.config.device_id]

            self._device = VDevice(params)
            logger.info("Hailo устройство создано: id=%d", self.config.device_id)

            # Загрузка модели
            self._load_model()

            self.is_initialized = True
            logger.info("Hailo инициализирован успешно")
            return True

        except Exception as e:
            logger.error("Ошибка инициализации Hailo: %s", e)
            self._hailo_available = False
            self.is_initialized = True  # Продолжаем работу без Hailo
            return False

    def shutdown(self) -> None:
        """Завершение работы и освобождение ресурсов."""
        if self._network_group:
            self._network_group.shutdown()

        if self._device:
            self._device.release()

        self._device = None
        self._hef = None
        self._network_group = None
        self.is_initialized = False
        self.model_loaded = False

        logger.info("Hailo завершил работу")

    def denoise(self, value: float) -> float:
        """Шумоподавление с помощью нейронной сети.

        Применяет обученную модель для фильтрации шума
        в сигнале магнитометра.

        Args:
            value: Входное значение в нТл

        Returns:
            float: Отфильтрованное значение
        """
        self._data_buffer.append(value)

        # Если модель не загружена или недостаточно данных, возвращаем как есть
        if not self.model_loaded or len(self._data_buffer) < self.BUFFER_SIZE:
            return value

        # Если Hailo недоступен, используем простой фильтр
        if not self._hailo_available:
            return self._simple_denoise(value)

        try:
            with self._inference_lock:
                # Подготовка входных данных
                input_data = np.array(
                    list(self._data_buffer),
                    dtype=np.float32,
                ).reshape(1, self.BUFFER_SIZE, 1)

                # Нормализация
                input_data = (input_data - self._baseline_mean) / max(
                    self._baseline_std, 1e-6
                )

                # Инференс
                output = self._run_inference(input_data)

                # Денормализация
                denoised = output[0, -1, 0] * self._baseline_std + self._baseline_mean
                return float(denoised)

        except Exception as e:
            logger.debug("Ошибка denoise, используем fallback: %s", e)
            return self._simple_denoise(value)

    def detect_anomaly(self, value: float) -> Tuple[bool, float]:
        """Обнаружение магнитной аномалии.

        Анализирует текущее значение и историю измерений
        для обнаружения аномалий.

        Args:
            value: Текущее значение в нТл

        Returns:
            Tuple[bool, float]: (обнаружена ли аномалия, уверенность)
        """
        # Обновление буфера
        self._data_buffer.append(value)

        # Накопление данных для базовой линии
        if len(self._samples_for_baseline) < 1000:
            self._samples_for_baseline.append(value)
            if len(self._samples_for_baseline) == 1000:
                self._update_baseline()
            return False, 0.0

        # Если недостаточно данных в буфере
        if len(self._data_buffer) < 32:
            return False, 0.0

        # Вычисление отклонения от базовой линии
        deviation = abs(value - self._baseline_mean) / max(self._baseline_std, 1e-6)

        # Быстрая проверка по порогу
        if deviation < 3.0:  # 3 sigma
            return False, 0.0

        # Детальный анализ с AI (если доступен)
        if self._hailo_available and self.model_loaded:
            detection = self._ai_detect_anomaly()
            return detection.detected, detection.confidence

        # Fallback: статистический метод
        confidence = min(1.0, (deviation - 3.0) / 5.0)  # Нормализация 3-8 sigma -> 0-1
        detected = confidence > self.ANOMALY_THRESHOLD

        return detected, confidence

    def classify_anomaly(self, window: List[float]) -> AnomalyDetection:
        """Классификация типа аномалии.

        Анализирует форму аномалии для определения её типа
        (металлический объект, геология, инфраструктура).

        Args:
            window: Окно данных с аномалией

        Returns:
            AnomalyDetection: Результат классификации
        """
        if len(window) < 10:
            return AnomalyDetection(
                detected=False,
                confidence=0.0,
                anomaly_type=AnomalyType.NONE,
                magnitude=0.0,
                description="Недостаточно данных",
            )

        arr = np.array(window)
        magnitude = float(np.max(np.abs(arr - np.mean(arr))))

        # Анализ формы сигнала
        # Дипольная аномалия (металлический объект): резкий пик
        # Геологическая: плавное изменение
        # Инфраструктура: периодический паттерн

        gradient = np.gradient(arr)
        max_gradient = float(np.max(np.abs(gradient)))

        # Простая эвристика для классификации
        if max_gradient > magnitude * 0.5:
            anomaly_type = AnomalyType.FERROUS_OBJECT
            description = "Возможно металлический объект (резкая дипольная аномалия)"
        elif self._check_periodic(arr):
            anomaly_type = AnomalyType.INFRASTRUCTURE
            description = "Возможно инфраструктура (периодический паттерн)"
        else:
            anomaly_type = AnomalyType.GEOLOGICAL
            description = "Возможно геологическая аномалия (плавное изменение)"

        return AnomalyDetection(
            detected=True,
            confidence=0.8,
            anomaly_type=anomaly_type,
            magnitude=magnitude,
            description=description,
        )

    def get_statistics(self) -> dict:
        """Получение статистики AI-процессора.

        Returns:
            dict: Статистика работы
        """
        return {
            "initialized": self.is_initialized,
            "hailo_available": getattr(self, "_hailo_available", False),
            "model_loaded": self.model_loaded,
            "buffer_size": len(self._data_buffer),
            "baseline_mean": self._baseline_mean,
            "baseline_std": self._baseline_std,
            "samples_collected": len(self._samples_for_baseline),
        }

    def _load_model(self) -> None:
        """Загрузка HEF-модели в Hailo."""
        model_path = Path(self.config.model_path)

        if not model_path.exists():
            logger.warning(
                "Файл модели не найден: %s, используется эмуляция",
                model_path,
            )
            self.model_loaded = False
            return

        try:
            from hailo_platform import HEF

            self._hef = HEF(str(model_path))

            # Конфигурация network group
            configure_params = self._device.create_configure_params(self._hef)
            self._network_group = self._device.configure(
                self._hef,
                configure_params,
            )[0]

            # Создание виртуальных потоков
            input_vstreams_params = self._network_group.make_input_vstream_params(
                quantized=False,
                format_type=FormatType.FLOAT32,
            )
            output_vstreams_params = self._network_group.make_output_vstream_params(
                quantized=False,
                format_type=FormatType.FLOAT32,
            )

            self._input_vstreams = self._network_group.create_input_vstreams(
                input_vstreams_params
            )
            self._output_vstreams = self._network_group.create_output_vstreams(
                output_vstreams_params
            )

            self.model_loaded = True
            logger.info("Модель загружена: %s", model_path)

        except Exception as e:
            logger.error("Ошибка загрузки модели: %s", e)
            self.model_loaded = False

    def _run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Запуск инференса на Hailo.

        Args:
            input_data: Входные данные

        Returns:
            np.ndarray: Результат инференса
        """
        if not self._hailo_available or not self.model_loaded:
            # Эмуляция: просто возвращаем сглаженные данные
            return input_data

        # Отправка данных
        for vstream in self._input_vstreams.values():
            vstream.send(input_data)

        # Получение результата
        results = {}
        for name, vstream in self._output_vstreams.items():
            results[name] = vstream.recv()

        # Возвращаем первый выход
        return list(results.values())[0]

    def _simple_denoise(self, value: float) -> float:
        """Простое шумоподавление без AI.

        Использует экспоненциальное скользящее среднее.

        Args:
            value: Входное значение

        Returns:
            float: Сглаженное значение
        """
        if not self._data_buffer:
            return value

        # EMA с alpha = 0.1
        alpha = 0.1
        if len(self._data_buffer) < 2:
            return value

        prev = self._data_buffer[-2]
        return alpha * value + (1 - alpha) * prev

    def _ai_detect_anomaly(self) -> AnomalyDetection:
        """AI-обнаружение аномалии.

        Returns:
            AnomalyDetection: Результат обнаружения
        """
        if len(self._data_buffer) < self.BUFFER_SIZE:
            return AnomalyDetection(
                detected=False,
                confidence=0.0,
                anomaly_type=AnomalyType.NONE,
                magnitude=0.0,
                description="Недостаточно данных",
            )

        try:
            with self._inference_lock:
                # Подготовка данных
                input_data = np.array(
                    list(self._data_buffer),
                    dtype=np.float32,
                ).reshape(1, self.BUFFER_SIZE, 1)

                # Нормализация
                input_data = (input_data - self._baseline_mean) / max(
                    self._baseline_std, 1e-6
                )

                # Инференс классификатора аномалий
                output = self._run_inference(input_data)

                # Интерпретация результата
                # Предполагаем выход [no_anomaly, ferrous, geological, infrastructure]
                if output.shape[-1] >= 4:
                    probs = output[0, -1, :]
                    confidence = 1 - probs[0]  # Вероятность аномалии
                    anomaly_idx = np.argmax(probs[1:]) + 1

                    anomaly_types = [
                        AnomalyType.NONE,
                        AnomalyType.FERROUS_OBJECT,
                        AnomalyType.GEOLOGICAL,
                        AnomalyType.INFRASTRUCTURE,
                    ]

                    return AnomalyDetection(
                        detected=confidence > self.ANOMALY_THRESHOLD,
                        confidence=float(confidence),
                        anomaly_type=anomaly_types[anomaly_idx],
                        magnitude=float(np.std(list(self._data_buffer)[-32:])),
                        description=f"AI-обнаружение: {anomaly_types[anomaly_idx].name}",
                    )

        except Exception as e:
            logger.debug("Ошибка AI-обнаружения: %s", e)

        # Fallback
        return AnomalyDetection(
            detected=False,
            confidence=0.0,
            anomaly_type=AnomalyType.NONE,
            magnitude=0.0,
            description="Ошибка анализа",
        )

    def _update_baseline(self) -> None:
        """Обновление базовой линии на основе собранных данных."""
        if self._samples_for_baseline:
            arr = np.array(self._samples_for_baseline)
            self._baseline_mean = float(np.mean(arr))
            self._baseline_std = float(np.std(arr))

            logger.info(
                "Базовая линия обновлена: mean=%.2f нТл, std=%.2f нТл",
                self._baseline_mean,
                self._baseline_std,
            )

    @staticmethod
    def _check_periodic(data: np.ndarray, threshold: float = 0.5) -> bool:
        """Проверка периодичности сигнала.

        Args:
            data: Массив данных
            threshold: Порог для определения периодичности

        Returns:
            bool: True если сигнал периодический
        """
        if len(data) < 32:
            return False

        # Автокорреляция
        data_centered = data - np.mean(data)
        autocorr = np.correlate(data_centered, data_centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]  # Нормализация

        # Поиск пиков (исключая нулевой лаг)
        peaks = []
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > threshold:
                    peaks.append(i)

        return len(peaks) >= 2
