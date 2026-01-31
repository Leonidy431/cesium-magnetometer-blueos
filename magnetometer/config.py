"""
Конфигурация модуля цезиевого магнитометра.

Содержит настройки подключения, параметры измерений и константы
для работы с цезиевым магнитометром на базе STM32.

Typical usage:
    >>> from magnetometer.config import MagnetometerConfig
    >>> config = MagnetometerConfig()
    >>> config.serial_port = "/dev/ttyUSB0"
    >>> config.sample_rate = 100

Author: НПО Лаборатория К
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class MeasurementMode(Enum):
    """Режимы измерения магнитометра.

    Attributes:
        CONTINUOUS: Непрерывное измерение с заданной частотой
        SINGLE: Одиночное измерение по запросу
        TRIGGERED: Измерение по внешнему триггеру
        GRADIENT: Режим градиентометра (два сенсора)
    """

    CONTINUOUS = auto()
    SINGLE = auto()
    TRIGGERED = auto()
    GRADIENT = auto()


class FilterType(Enum):
    """Типы цифровых фильтров.

    Attributes:
        NONE: Без фильтрации
        LOWPASS: Фильтр низких частот
        BANDPASS: Полосовой фильтр
        KALMAN: Фильтр Калмана
        AI_DENOISE: AI-шумоподавление через Hailo
    """

    NONE = auto()
    LOWPASS = auto()
    BANDPASS = auto()
    KALMAN = auto()
    AI_DENOISE = auto()


@dataclass
class SerialConfig:
    """Конфигурация последовательного порта.

    Attributes:
        port: Путь к последовательному порту (например, /dev/ttyUSB0)
        baudrate: Скорость передачи данных в бодах
        timeout: Таймаут чтения в секундах
        parity: Бит чётности ('N', 'E', 'O')
        stopbits: Количество стоп-битов (1, 1.5, 2)
        bytesize: Размер байта данных (5-8)
    """

    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 1.0
    parity: str = "N"
    stopbits: float = 1
    bytesize: int = 8


@dataclass
class HailoConfig:
    """Конфигурация AI-ускорителя Hailo 8L.

    Attributes:
        device_id: Идентификатор устройства Hailo
        model_path: Путь к HEF-файлу модели
        batch_size: Размер батча для инференса
        power_mode: Режим энергопотребления (0-3)
        enabled: Флаг включения AI-обработки
    """

    device_id: int = 0
    model_path: str = "/opt/magnit/models/magnetometer_denoise.hef"
    batch_size: int = 32
    power_mode: int = 1
    enabled: bool = True


@dataclass
class BlueOSConfig:
    """Конфигурация интеграции с BlueOS.

    Attributes:
        host: Хост BlueOS API
        port: Порт BlueOS API
        api_version: Версия API
        extension_name: Имя расширения в BlueOS
        mavlink_system_id: System ID для MAVLink
        mavlink_component_id: Component ID для MAVLink
    """

    host: str = "localhost"
    port: int = 6040
    api_version: str = "v1"
    extension_name: str = "cesium-magnetometer"
    mavlink_system_id: int = 1
    mavlink_component_id: int = 191


@dataclass
class CalibrationData:
    """Данные калибровки магнитометра.

    Attributes:
        offset_x: Смещение по оси X в нТл
        offset_y: Смещение по оси Y в нТл
        offset_z: Смещение по оси Z в нТл
        scale_x: Масштабный коэффициент по оси X
        scale_y: Масштабный коэффициент по оси Y
        scale_z: Масштабный коэффициент по оси Z
        temperature_coefficient: Температурный коэффициент нТл/°C
        last_calibration: Дата последней калибровки (ISO формат)
    """

    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    temperature_coefficient: float = 0.0
    last_calibration: Optional[str] = None


@dataclass
class MagnetometerConfig:
    """Главная конфигурация модуля магнитометра.

    Объединяет все настройки для работы цезиевого магнитометра
    с BlueOS и AI-обработкой через Hailo 8L.

    Attributes:
        serial: Настройки последовательного порта
        hailo: Настройки AI-ускорителя
        blueos: Настройки интеграции с BlueOS
        calibration: Данные калибровки
        sample_rate: Частота измерений в Гц (1-1000)
        measurement_mode: Режим измерения
        filter_type: Тип цифрового фильтра
        filter_cutoff: Частота среза фильтра в Гц
        averaging_samples: Количество выборок для усреднения
        log_level: Уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        data_directory: Директория для сохранения данных

    Example:
        >>> config = MagnetometerConfig()
        >>> config.sample_rate = 100
        >>> config.measurement_mode = MeasurementMode.CONTINUOUS
        >>> config.hailo.enabled = True
    """

    # Вложенные конфигурации
    serial: SerialConfig = field(default_factory=SerialConfig)
    hailo: HailoConfig = field(default_factory=HailoConfig)
    blueos: BlueOSConfig = field(default_factory=BlueOSConfig)
    calibration: CalibrationData = field(default_factory=CalibrationData)

    # Параметры измерений
    sample_rate: int = 100
    measurement_mode: MeasurementMode = MeasurementMode.CONTINUOUS
    filter_type: FilterType = FilterType.KALMAN
    filter_cutoff: float = 10.0
    averaging_samples: int = 10

    # Системные настройки
    log_level: str = "INFO"
    data_directory: str = "/var/log/magnit/magnetometer"

    # Константы цезиевого магнитометра
    GYROMAGNETIC_RATIO: float = 3.498  # Гц/нТл для Cs-133
    MIN_FIELD: float = 20000.0  # Минимальное поле в нТл
    MAX_FIELD: float = 100000.0  # Максимальное поле в нТл
    RESOLUTION: float = 0.001  # Разрешение в нТл

    def validate(self) -> bool:
        """Проверка корректности конфигурации.

        Returns:
            bool: True если конфигурация валидна

        Raises:
            ValueError: Если параметры выходят за допустимые пределы
        """
        if not 1 <= self.sample_rate <= 1000:
            raise ValueError(
                f"sample_rate должен быть в диапазоне 1-1000, "
                f"получено: {self.sample_rate}"
            )

        if self.filter_cutoff <= 0:
            raise ValueError(
                f"filter_cutoff должен быть положительным, "
                f"получено: {self.filter_cutoff}"
            )

        if self.averaging_samples < 1:
            raise ValueError(
                f"averaging_samples должен быть >= 1, "
                f"получено: {self.averaging_samples}"
            )

        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError(
                f"Недопустимый log_level: {self.log_level}"
            )

        return True

    def to_dict(self) -> dict:
        """Преобразование конфигурации в словарь.

        Returns:
            dict: Словарь с параметрами конфигурации
        """
        return {
            "serial": {
                "port": self.serial.port,
                "baudrate": self.serial.baudrate,
                "timeout": self.serial.timeout,
            },
            "hailo": {
                "enabled": self.hailo.enabled,
                "model_path": self.hailo.model_path,
                "batch_size": self.hailo.batch_size,
            },
            "blueos": {
                "host": self.blueos.host,
                "port": self.blueos.port,
                "extension_name": self.blueos.extension_name,
            },
            "measurement": {
                "sample_rate": self.sample_rate,
                "mode": self.measurement_mode.name,
                "filter": self.filter_type.name,
            },
        }
