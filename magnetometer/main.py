#!/usr/bin/env python3
"""
Точка входа для запуска расширения BlueOS магнитометра.

Запуск:
    python -m magnetometer.main
    # или
    magnit-magnetometer

Аргументы командной строки:
    --port PORT     Последовательный порт STM32 (default: /dev/ttyUSB0)
    --baudrate N    Скорость порта (default: 115200)
    --sample-rate N Частота измерений в Гц (default: 100)
    --no-hailo      Отключить AI-обработку Hailo
    --web-port N    Порт веб-сервера (default: 9090)
    --log-level LVL Уровень логирования (default: INFO)

Author: НПО Лаборатория К
"""

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

from .blueos_extension import BlueOSExtension, run_extension
from .cesium_magnetometer import CesiumMagnetometer
from .config import MagnetometerConfig, SerialConfig


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов командной строки.

    Returns:
        argparse.Namespace: Разобранные аргументы
    """
    parser = argparse.ArgumentParser(
        description="Cesium Magnetometer BlueOS Extension",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Последовательный порт STM32",
    )

    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Скорость последовательного порта",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=100,
        help="Частота измерений в Гц (1-1000)",
    )

    parser.add_argument(
        "--no-hailo",
        action="store_true",
        help="Отключить AI-обработку Hailo",
    )

    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Порт веб-сервера BlueOS расширения",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования",
    )

    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="Показать доступные последовательные порты и выйти",
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Настройка логирования.

    Args:
        level: Уровень логирования
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def list_serial_ports() -> None:
    """Вывод списка доступных последовательных портов."""
    from .stm32_driver import STM32Driver

    ports = STM32Driver.list_ports()

    if not ports:
        print("Последовательные порты не найдены")
        return

    print("Доступные последовательные порты:")
    print("-" * 60)

    for port, desc, hwid in ports:
        print(f"  {port}")
        print(f"    Описание: {desc}")
        print(f"    HWID: {hwid}")
        print()


def create_config(args: argparse.Namespace) -> MagnetometerConfig:
    """Создание конфигурации из аргументов.

    Args:
        args: Аргументы командной строки

    Returns:
        MagnetometerConfig: Конфигурация магнитометра
    """
    config = MagnetometerConfig()

    # Serial
    config.serial.port = args.port
    config.serial.baudrate = args.baudrate

    # Measurement
    config.sample_rate = args.sample_rate

    # Hailo
    config.hailo.enabled = not args.no_hailo

    # Logging
    config.log_level = args.log_level

    return config


async def async_main(args: argparse.Namespace) -> int:
    """Асинхронная точка входа.

    Args:
        args: Аргументы командной строки

    Returns:
        int: Код возврата (0 = успех)
    """
    # Создание конфигурации
    config = create_config(args)

    # Создание магнитометра
    magnetometer = CesiumMagnetometer(config)

    # Обработка сигналов
    stop_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info("Получен сигнал %s, завершение...", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Подключение к магнитометру
        logger.info("Подключение к магнитометру на %s...", args.port)
        magnetometer.connect()

        # Создание и запуск расширения BlueOS
        extension = BlueOSExtension(magnetometer)
        extension.WEB_PORT = args.web_port

        await extension.start()

        logger.info("=" * 60)
        logger.info("Cesium Magnetometer BlueOS Extension запущен")
        logger.info("  Web UI: http://localhost:%d", args.web_port)
        logger.info("  API: http://localhost:%d/v1", args.web_port)
        logger.info("  WebSocket: ws://localhost:%d/v1/ws", args.web_port)
        logger.info("=" * 60)

        # Ожидание сигнала остановки
        await stop_event.wait()

    except ConnectionError as e:
        logger.error("Ошибка подключения: %s", e)
        return 1

    except Exception as e:
        logger.exception("Неожиданная ошибка: %s", e)
        return 1

    finally:
        # Остановка
        logger.info("Остановка сервисов...")

        if "extension" in locals():
            await extension.stop()

        magnetometer.disconnect()
        logger.info("Завершено")

    return 0


def main() -> int:
    """Главная точка входа.

    Returns:
        int: Код возврата
    """
    args = parse_args()

    # Настройка логирования
    setup_logging(args.log_level)

    # Список портов
    if args.list_ports:
        list_serial_ports()
        return 0

    # Запуск асинхронного main
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
