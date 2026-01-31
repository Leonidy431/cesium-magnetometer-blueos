"""
Модуль цезиевого магнитометра для BlueOS.

Интеграция цезиевого магнитометра на базе STM32 с системой BlueOS
для подводных роботов с поддержкой AI-обработки через Hailo 8L.

Modules:
    cesium_magnetometer: Основной класс магнитометра
    stm32_driver: Драйвер связи с STM32
    blueos_extension: Расширение для BlueOS
    hailo_processor: AI-обработка данных через Hailo 8L
    config: Конфигурация модуля

Example:
    >>> from magnetometer import CesiumMagnetometer
    >>> mag = CesiumMagnetometer()
    >>> mag.connect()
    >>> reading = mag.get_reading()
    >>> print(f"Magnetic field: {reading.value} nT")

Author: НПО Лаборатория К
Version: 1.0.0
License: Proprietary
"""

__version__ = "1.0.0"
__author__ = "НПО Лаборатория К"
__email__ = "lab767@gmail.com"

from .cesium_magnetometer import CesiumMagnetometer
from .stm32_driver import STM32Driver
from .blueos_extension import BlueOSExtension
from .hailo_processor import HailoProcessor
from .config import MagnetometerConfig

__all__ = [
    "CesiumMagnetometer",
    "STM32Driver",
    "BlueOSExtension",
    "HailoProcessor",
    "MagnetometerConfig",
]
