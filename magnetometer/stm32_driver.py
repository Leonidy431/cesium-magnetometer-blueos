"""
Драйвер связи с STM32 для цезиевого магнитометра.

Обеспечивает низкоуровневую связь с микроконтроллером STM32,
который управляет цезиевым магнитометром. Поддерживает протокол
обмена данными, команды управления и диагностику.

Protocol:
    Формат пакета: [START][LEN][CMD][DATA...][CRC16]
    START: 0xAA 0x55
    LEN: длина данных (1 байт)
    CMD: код команды (1 байт)
    DATA: данные команды (0-255 байт)
    CRC16: контрольная сумма (2 байта, little-endian)

Author: НПО Лаборатория К
"""

import logging
import struct
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional, Tuple

import serial
from serial.tools import list_ports

from .config import SerialConfig


logger = logging.getLogger(__name__)


class STM32Command(IntEnum):
    """Коды команд протокола STM32.

    Attributes:
        PING: Проверка связи
        GET_STATUS: Запрос статуса устройства
        GET_READING: Запрос текущего измерения
        SET_SAMPLE_RATE: Установка частоты измерений
        START_CONTINUOUS: Запуск непрерывного измерения
        STOP_CONTINUOUS: Остановка непрерывного измерения
        CALIBRATE: Запуск калибровки
        GET_CALIBRATION: Запрос данных калибровки
        SET_CALIBRATION: Установка данных калибровки
        RESET: Сброс устройства
        GET_FIRMWARE_VERSION: Запрос версии прошивки
        SET_FILTER: Установка параметров фильтра
        GET_DIAGNOSTICS: Запрос диагностических данных
    """

    PING = 0x01
    GET_STATUS = 0x02
    GET_READING = 0x03
    SET_SAMPLE_RATE = 0x04
    START_CONTINUOUS = 0x05
    STOP_CONTINUOUS = 0x06
    CALIBRATE = 0x10
    GET_CALIBRATION = 0x11
    SET_CALIBRATION = 0x12
    RESET = 0x20
    GET_FIRMWARE_VERSION = 0x21
    SET_FILTER = 0x30
    GET_DIAGNOSTICS = 0x40


class STM32Status(IntEnum):
    """Коды статуса устройства STM32.

    Attributes:
        OK: Устройство работает нормально
        INITIALIZING: Устройство инициализируется
        CALIBRATING: Выполняется калибровка
        ERROR_SENSOR: Ошибка сенсора
        ERROR_TEMPERATURE: Ошибка температуры
        ERROR_COMMUNICATION: Ошибка связи
        LOW_SIGNAL: Низкий уровень сигнала
    """

    OK = 0x00
    INITIALIZING = 0x01
    CALIBRATING = 0x02
    ERROR_SENSOR = 0x80
    ERROR_TEMPERATURE = 0x81
    ERROR_COMMUNICATION = 0x82
    LOW_SIGNAL = 0x83


@dataclass
class MagnetometerReading:
    """Структура данных измерения магнитометра.

    Attributes:
        timestamp: Временная метка измерения (мс с момента старта)
        field_total: Полное магнитное поле в нТл
        field_x: Компонента X в нТл (если доступно)
        field_y: Компонента Y в нТл (если доступно)
        field_z: Компонента Z в нТл (если доступно)
        temperature: Температура сенсора в °C
        signal_quality: Качество сигнала (0-100%)
        status: Статус измерения
    """

    timestamp: int
    field_total: float
    field_x: Optional[float] = None
    field_y: Optional[float] = None
    field_z: Optional[float] = None
    temperature: float = 25.0
    signal_quality: int = 100
    status: STM32Status = STM32Status.OK


@dataclass
class DiagnosticsData:
    """Диагностические данные устройства.

    Attributes:
        uptime_seconds: Время работы устройства в секундах
        cpu_temperature: Температура процессора STM32 в °C
        voltage_3v3: Напряжение питания 3.3В
        voltage_5v: Напряжение питания 5В
        error_count: Количество ошибок с момента старта
        readings_count: Количество измерений
        last_error_code: Код последней ошибки
    """

    uptime_seconds: int
    cpu_temperature: float
    voltage_3v3: float
    voltage_5v: float
    error_count: int
    readings_count: int
    last_error_code: int


class STM32Driver:
    """Драйвер связи с STM32 микроконтроллером.

    Обеспечивает связь с STM32 по последовательному порту,
    отправку команд, приём данных и обработку ошибок.

    Attributes:
        config: Конфигурация последовательного порта
        is_connected: Флаг состояния подключения
        firmware_version: Версия прошивки STM32

    Example:
        >>> driver = STM32Driver(SerialConfig(port="/dev/ttyUSB0"))
        >>> driver.connect()
        >>> reading = driver.get_reading()
        >>> print(f"Field: {reading.field_total} nT")
        >>> driver.disconnect()
    """

    # Константы протокола
    START_BYTES = bytes([0xAA, 0x55])
    RESPONSE_TIMEOUT = 1.0  # секунды
    MAX_PACKET_SIZE = 260  # START(2) + LEN(1) + CMD(1) + DATA(255) + CRC(2)

    def __init__(self, config: Optional[SerialConfig] = None):
        """Инициализация драйвера STM32.

        Args:
            config: Конфигурация последовательного порта.
                    Если не указана, используются значения по умолчанию.
        """
        self.config = config or SerialConfig()
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._continuous_callback: Optional[Callable[[MagnetometerReading], None]] = None
        self._continuous_thread: Optional[threading.Thread] = None
        self._stop_continuous = threading.Event()

        self.is_connected: bool = False
        self.firmware_version: str = "unknown"

        logger.debug(
            "STM32Driver инициализирован: port=%s, baudrate=%d",
            self.config.port,
            self.config.baudrate,
        )

    def connect(self) -> bool:
        """Установка соединения с STM32.

        Открывает последовательный порт и проверяет связь
        с устройством командой PING.

        Returns:
            bool: True если соединение установлено успешно

        Raises:
            serial.SerialException: Ошибка открытия порта
            TimeoutError: Устройство не отвечает
        """
        if self.is_connected:
            logger.warning("Соединение уже установлено")
            return True

        try:
            self._serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                parity=self.config.parity,
                stopbits=self.config.stopbits,
                bytesize=self.config.bytesize,
            )

            # Очистка буферов
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            # Небольшая задержка для инициализации
            time.sleep(0.1)

            # Проверка связи
            if not self._ping():
                raise TimeoutError("STM32 не отвечает на PING")

            # Получение версии прошивки
            self.firmware_version = self._get_firmware_version()

            self.is_connected = True
            logger.info(
                "Соединение с STM32 установлено: firmware=%s",
                self.firmware_version,
            )
            return True

        except serial.SerialException as e:
            logger.error("Ошибка открытия порта %s: %s", self.config.port, e)
            raise

    def disconnect(self) -> None:
        """Закрытие соединения с STM32.

        Останавливает непрерывное измерение (если запущено)
        и закрывает последовательный порт.
        """
        if self._continuous_thread and self._continuous_thread.is_alive():
            self.stop_continuous()

        if self._serial and self._serial.is_open:
            self._serial.close()

        self._serial = None
        self.is_connected = False
        logger.info("Соединение с STM32 закрыто")

    def get_reading(self) -> MagnetometerReading:
        """Получение текущего измерения магнитометра.

        Returns:
            MagnetometerReading: Структура с данными измерения

        Raises:
            ConnectionError: Соединение не установлено
            TimeoutError: Таймаут ожидания ответа
            ValueError: Ошибка разбора ответа
        """
        self._check_connection()

        response = self._send_command(STM32Command.GET_READING)
        return self._parse_reading(response)

    def get_status(self) -> STM32Status:
        """Получение текущего статуса устройства.

        Returns:
            STM32Status: Код статуса устройства

        Raises:
            ConnectionError: Соединение не установлено
        """
        self._check_connection()

        response = self._send_command(STM32Command.GET_STATUS)
        if len(response) < 1:
            raise ValueError("Пустой ответ на GET_STATUS")

        return STM32Status(response[0])

    def set_sample_rate(self, rate_hz: int) -> bool:
        """Установка частоты измерений.

        Args:
            rate_hz: Частота измерений в Гц (1-1000)

        Returns:
            bool: True если команда выполнена успешно

        Raises:
            ValueError: Частота вне допустимого диапазона
        """
        if not 1 <= rate_hz <= 1000:
            raise ValueError(f"Частота должна быть 1-1000 Гц, получено: {rate_hz}")

        self._check_connection()

        data = struct.pack("<H", rate_hz)  # Little-endian uint16
        response = self._send_command(STM32Command.SET_SAMPLE_RATE, data)

        success = len(response) > 0 and response[0] == 0x00
        if success:
            logger.info("Частота измерений установлена: %d Гц", rate_hz)
        else:
            logger.error("Ошибка установки частоты измерений")

        return success

    def start_continuous(
        self,
        callback: Callable[[MagnetometerReading], None],
    ) -> bool:
        """Запуск непрерывного измерения.

        Args:
            callback: Функция обратного вызова для получения данных.
                      Вызывается для каждого нового измерения.

        Returns:
            bool: True если измерение запущено

        Example:
            >>> def on_reading(reading):
            ...     print(f"Field: {reading.field_total} nT")
            >>> driver.start_continuous(on_reading)
        """
        self._check_connection()

        if self._continuous_thread and self._continuous_thread.is_alive():
            logger.warning("Непрерывное измерение уже запущено")
            return False

        response = self._send_command(STM32Command.START_CONTINUOUS)
        if len(response) == 0 or response[0] != 0x00:
            logger.error("Ошибка запуска непрерывного измерения")
            return False

        self._continuous_callback = callback
        self._stop_continuous.clear()
        self._continuous_thread = threading.Thread(
            target=self._continuous_reader,
            daemon=True,
        )
        self._continuous_thread.start()

        logger.info("Непрерывное измерение запущено")
        return True

    def stop_continuous(self) -> bool:
        """Остановка непрерывного измерения.

        Returns:
            bool: True если измерение остановлено
        """
        self._check_connection()

        self._stop_continuous.set()

        response = self._send_command(STM32Command.STOP_CONTINUOUS)
        if len(response) == 0 or response[0] != 0x00:
            logger.error("Ошибка остановки непрерывного измерения")
            return False

        if self._continuous_thread:
            self._continuous_thread.join(timeout=2.0)
            self._continuous_thread = None

        self._continuous_callback = None
        logger.info("Непрерывное измерение остановлено")
        return True

    def calibrate(self, timeout_seconds: float = 60.0) -> bool:
        """Запуск процедуры калибровки.

        Калибровка занимает до 60 секунд. Во время калибровки
        устройство должно находиться в стабильном магнитном поле.

        Args:
            timeout_seconds: Максимальное время ожидания калибровки

        Returns:
            bool: True если калибровка успешна
        """
        self._check_connection()

        response = self._send_command(
            STM32Command.CALIBRATE,
            timeout=timeout_seconds,
        )

        success = len(response) > 0 and response[0] == 0x00
        if success:
            logger.info("Калибровка завершена успешно")
        else:
            logger.error("Ошибка калибровки")

        return success

    def get_diagnostics(self) -> DiagnosticsData:
        """Получение диагностических данных устройства.

        Returns:
            DiagnosticsData: Структура с диагностическими данными
        """
        self._check_connection()

        response = self._send_command(STM32Command.GET_DIAGNOSTICS)

        if len(response) < 20:
            raise ValueError(
                f"Недостаточно данных в ответе: {len(response)} байт"
            )

        # Распаковка данных: I=uint32, f=float, H=uint16
        uptime, cpu_temp, v33, v5, errors, readings, last_error = struct.unpack(
            "<IffHIIH",
            response[:22],
        )

        return DiagnosticsData(
            uptime_seconds=uptime,
            cpu_temperature=cpu_temp,
            voltage_3v3=v33,
            voltage_5v=v5,
            error_count=errors,
            readings_count=readings,
            last_error_code=last_error,
        )

    def reset(self) -> None:
        """Сброс устройства STM32.

        После сброса потребуется повторное подключение.
        """
        self._check_connection()

        try:
            self._send_command(STM32Command.RESET, timeout=0.1)
        except TimeoutError:
            pass  # Ожидаемо - устройство перезагружается

        self.is_connected = False
        if self._serial:
            self._serial.close()
        self._serial = None

        logger.info("STM32 сброшен, требуется повторное подключение")

    @staticmethod
    def list_ports() -> list:
        """Получение списка доступных последовательных портов.

        Returns:
            list: Список кортежей (порт, описание, hwid)

        Example:
            >>> for port, desc, hwid in STM32Driver.list_ports():
            ...     print(f"{port}: {desc}")
        """
        return [(p.device, p.description, p.hwid) for p in list_ports.comports()]

    def _check_connection(self) -> None:
        """Проверка состояния соединения.

        Raises:
            ConnectionError: Соединение не установлено
        """
        if not self.is_connected or not self._serial or not self._serial.is_open:
            raise ConnectionError("Соединение с STM32 не установлено")

    def _send_command(
        self,
        command: STM32Command,
        data: bytes = b"",
        timeout: Optional[float] = None,
    ) -> bytes:
        """Отправка команды и получение ответа.

        Args:
            command: Код команды
            data: Данные команды (опционально)
            timeout: Таймаут ожидания ответа

        Returns:
            bytes: Данные ответа (без заголовка и CRC)

        Raises:
            TimeoutError: Таймаут ожидания ответа
            ValueError: Ошибка CRC или формата пакета
        """
        timeout = timeout or self.RESPONSE_TIMEOUT

        # Формирование пакета
        packet = self._build_packet(command, data)

        with self._lock:
            # Очистка входного буфера
            self._serial.reset_input_buffer()

            # Отправка
            self._serial.write(packet)
            self._serial.flush()

            # Приём ответа
            response = self._receive_packet(timeout)

        return response

    def _build_packet(self, command: STM32Command, data: bytes) -> bytes:
        """Формирование пакета для отправки.

        Args:
            command: Код команды
            data: Данные команды

        Returns:
            bytes: Сформированный пакет
        """
        length = len(data)
        if length > 255:
            raise ValueError(f"Данные слишком длинные: {length} > 255")

        # Заголовок + длина + команда + данные
        packet = self.START_BYTES + bytes([length, command]) + data

        # CRC16 (CCITT)
        crc = self._calculate_crc16(packet[2:])  # CRC от LEN+CMD+DATA
        packet += struct.pack("<H", crc)

        return packet

    def _receive_packet(self, timeout: float) -> bytes:
        """Приём и разбор пакета ответа.

        Args:
            timeout: Таймаут ожидания

        Returns:
            bytes: Данные ответа

        Raises:
            TimeoutError: Таймаут ожидания
            ValueError: Ошибка формата или CRC
        """
        start_time = time.time()
        buffer = bytearray()

        # Ожидание стартовых байтов
        while time.time() - start_time < timeout:
            if self._serial.in_waiting > 0:
                byte = self._serial.read(1)
                buffer.append(byte[0])

                if len(buffer) >= 2:
                    if buffer[-2:] == bytearray(self.START_BYTES):
                        buffer = bytearray(self.START_BYTES)
                        break
            else:
                time.sleep(0.001)
        else:
            raise TimeoutError("Таймаут ожидания стартовых байтов")

        # Чтение длины и команды
        header = self._serial.read(2)
        if len(header) < 2:
            raise TimeoutError("Таймаут чтения заголовка")

        length = header[0]
        # command = header[1]  # Можно использовать для проверки

        # Чтение данных
        if length > 0:
            data = self._serial.read(length)
            if len(data) < length:
                raise TimeoutError("Таймаут чтения данных")
        else:
            data = b""

        # Чтение CRC
        crc_bytes = self._serial.read(2)
        if len(crc_bytes) < 2:
            raise TimeoutError("Таймаут чтения CRC")

        received_crc = struct.unpack("<H", crc_bytes)[0]

        # Проверка CRC
        calculated_crc = self._calculate_crc16(header + data)
        if received_crc != calculated_crc:
            raise ValueError(
                f"Ошибка CRC: получено 0x{received_crc:04X}, "
                f"вычислено 0x{calculated_crc:04X}"
            )

        return bytes(data)

    @staticmethod
    def _calculate_crc16(data: bytes) -> int:
        """Вычисление CRC16-CCITT.

        Args:
            data: Данные для расчёта CRC

        Returns:
            int: Значение CRC16
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc

    def _ping(self) -> bool:
        """Проверка связи с устройством.

        Returns:
            bool: True если устройство отвечает
        """
        try:
            response = self._send_command(STM32Command.PING)
            return len(response) > 0 and response[0] == 0x00
        except (TimeoutError, ValueError):
            return False

    def _get_firmware_version(self) -> str:
        """Получение версии прошивки.

        Returns:
            str: Строка версии (например, "1.2.3")
        """
        try:
            response = self._send_command(STM32Command.GET_FIRMWARE_VERSION)
            if len(response) >= 3:
                major, minor, patch = response[0], response[1], response[2]
                return f"{major}.{minor}.{patch}"
        except (TimeoutError, ValueError):
            pass
        return "unknown"

    def _parse_reading(self, data: bytes) -> MagnetometerReading:
        """Разбор данных измерения.

        Args:
            data: Сырые данные измерения

        Returns:
            MagnetometerReading: Структура измерения
        """
        if len(data) < 16:
            raise ValueError(f"Недостаточно данных: {len(data)} < 16 байт")

        # Формат: timestamp(4) + field_total(4) + temperature(4) +
        #         signal_quality(1) + status(1) + reserved(2)
        timestamp, field_total, temperature, quality, status = struct.unpack(
            "<IffBBxx",
            data[:16],
        )

        reading = MagnetometerReading(
            timestamp=timestamp,
            field_total=field_total,
            temperature=temperature,
            signal_quality=quality,
            status=STM32Status(status),
        )

        # Опциональные компоненты X, Y, Z
        if len(data) >= 28:
            field_x, field_y, field_z = struct.unpack("<fff", data[16:28])
            reading.field_x = field_x
            reading.field_y = field_y
            reading.field_z = field_z

        return reading

    def _continuous_reader(self) -> None:
        """Поток чтения данных в режиме непрерывного измерения."""
        logger.debug("Поток непрерывного чтения запущен")

        while not self._stop_continuous.is_set():
            try:
                if self._serial and self._serial.in_waiting >= 20:
                    data = self._receive_packet(timeout=0.1)
                    reading = self._parse_reading(data)

                    if self._continuous_callback:
                        try:
                            self._continuous_callback(reading)
                        except Exception as e:
                            logger.error("Ошибка в callback: %s", e)
                else:
                    time.sleep(0.001)

            except TimeoutError:
                continue
            except Exception as e:
                logger.error("Ошибка чтения данных: %s", e)
                time.sleep(0.01)

        logger.debug("Поток непрерывного чтения остановлен")
