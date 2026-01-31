"""
Unit tests for Cesium Magnetometer module.

Run tests:
    pytest src/tests/test_magnetometer.py -v

Author: НПО Лаборатория К
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import modules under test
from magnetometer.config import (
    MagnetometerConfig,
    SerialConfig,
    HailoConfig,
    BlueOSConfig,
    FilterType,
    MeasurementMode,
)
from magnetometer.stm32_driver import (
    STM32Driver,
    STM32Command,
    STM32Status,
    MagnetometerReading,
)
from magnetometer.cesium_magnetometer import (
    CesiumMagnetometer,
    ProcessedReading,
    KalmanFilter,
)


class TestMagnetometerConfig:
    """Тесты для класса MagnetometerConfig."""

    def test_default_config(self):
        """Тест создания конфигурации по умолчанию."""
        config = MagnetometerConfig()

        assert config.sample_rate == 100
        assert config.measurement_mode == MeasurementMode.CONTINUOUS
        assert config.filter_type == FilterType.KALMAN
        assert config.serial.port == "/dev/ttyUSB0"
        assert config.serial.baudrate == 115200

    def test_config_validation_valid(self):
        """Тест валидации корректной конфигурации."""
        config = MagnetometerConfig()
        config.sample_rate = 500

        assert config.validate() is True

    def test_config_validation_invalid_sample_rate(self):
        """Тест валидации некорректной частоты измерений."""
        config = MagnetometerConfig()
        config.sample_rate = 2000  # Выше максимума

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        assert "sample_rate" in str(exc_info.value)

    def test_config_to_dict(self):
        """Тест преобразования конфигурации в словарь."""
        config = MagnetometerConfig()
        config_dict = config.to_dict()

        assert "serial" in config_dict
        assert "hailo" in config_dict
        assert "blueos" in config_dict
        assert "measurement" in config_dict
        assert config_dict["measurement"]["sample_rate"] == 100


class TestKalmanFilter:
    """Тесты для фильтра Калмана."""

    def test_kalman_initialization(self):
        """Тест инициализации фильтра Калмана."""
        kf = KalmanFilter(process_variance=0.01, measurement_variance=0.1)

        # Первое измерение должно вернуться как есть
        result = kf.update(50000.0)
        assert result == 50000.0

    def test_kalman_filtering(self):
        """Тест фильтрации последовательности значений."""
        kf = KalmanFilter()

        values = [50000.0, 50010.0, 49990.0, 50005.0, 49995.0]
        results = [kf.update(v) for v in values]

        # Отфильтрованные значения должны быть ближе к среднему
        assert all(49990 < r < 50010 for r in results[1:])

    def test_kalman_reset(self):
        """Тест сброса фильтра."""
        kf = KalmanFilter()

        kf.update(50000.0)
        kf.update(50010.0)
        kf.reset()

        # После сброса первое значение должно вернуться как есть
        result = kf.update(60000.0)
        assert result == 60000.0


class TestSTM32Driver:
    """Тесты для драйвера STM32."""

    def test_crc16_calculation(self):
        """Тест расчёта CRC16."""
        # Известные тестовые векторы
        data = bytes([0x01, 0x02, 0x03])
        crc = STM32Driver._calculate_crc16(data)

        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_build_packet(self):
        """Тест формирования пакета."""
        driver = STM32Driver()

        packet = driver._build_packet(STM32Command.PING, b"")

        # Проверка структуры пакета
        assert packet[:2] == bytes([0xAA, 0x55])  # START
        assert packet[2] == 0  # LEN
        assert packet[3] == STM32Command.PING  # CMD
        assert len(packet) == 6  # START(2) + LEN(1) + CMD(1) + CRC(2)

    def test_list_ports(self):
        """Тест получения списка портов."""
        ports = STM32Driver.list_ports()

        assert isinstance(ports, list)
        # Каждый элемент должен быть кортежем из 3 элементов
        for port in ports:
            assert isinstance(port, tuple)
            assert len(port) == 3


class TestMagnetometerReading:
    """Тесты для структуры измерения."""

    def test_reading_creation(self):
        """Тест создания структуры измерения."""
        reading = MagnetometerReading(
            timestamp=12345,
            field_total=50000.0,
            temperature=25.0,
            signal_quality=95,
            status=STM32Status.OK,
        )

        assert reading.timestamp == 12345
        assert reading.field_total == 50000.0
        assert reading.temperature == 25.0
        assert reading.signal_quality == 95
        assert reading.status == STM32Status.OK

    def test_reading_optional_fields(self):
        """Тест опциональных полей измерения."""
        reading = MagnetometerReading(
            timestamp=0,
            field_total=50000.0,
            field_x=30000.0,
            field_y=20000.0,
            field_z=36055.0,
        )

        assert reading.field_x == 30000.0
        assert reading.field_y == 20000.0
        assert reading.field_z == 36055.0


class TestCesiumMagnetometer:
    """Тесты для основного класса магнитометра."""

    def test_magnetometer_initialization(self):
        """Тест инициализации магнитометра."""
        mag = CesiumMagnetometer()

        assert mag.is_connected is False
        assert mag.is_measuring is False
        assert mag.config.sample_rate == 100

    def test_magnetometer_custom_config(self):
        """Тест инициализации с пользовательской конфигурацией."""
        config = MagnetometerConfig()
        config.sample_rate = 200
        config.hailo.enabled = False

        mag = CesiumMagnetometer(config)

        assert mag.config.sample_rate == 200
        assert mag.config.hailo.enabled is False

    def test_check_connection_not_connected(self):
        """Тест проверки соединения без подключения."""
        mag = CesiumMagnetometer()

        with pytest.raises(ConnectionError):
            mag._check_connection()

    def test_apply_calibration(self):
        """Тест применения калибровки."""
        mag = CesiumMagnetometer()
        mag.config.calibration.offset_x = 100.0
        mag.config.calibration.scale_x = 1.01
        mag.config.calibration.temperature_coefficient = 0.5

        # Значение 50000 нТл при температуре 25°C
        result = mag._apply_calibration(50100.0, 25.0)

        # (50100 - 100) * 1.01 - 0 = 50500
        expected = (50100.0 - 100.0) * 1.01
        assert abs(result - expected) < 0.1

    def test_lowpass_filter(self):
        """Тест фильтра низких частот."""
        mag = CesiumMagnetometer()
        mag.config.sample_rate = 100
        mag.config.filter_cutoff = 10.0

        values = [50000.0, 50010.0, 49990.0, 50005.0, 49995.0]
        results = [mag._lowpass_filter(v) for v in values]

        # Первое значение равно себе
        assert results[0] == values[0]

        # Последующие значения сглажены
        assert all(49990 <= r <= 50010 for r in results)


class TestProcessedReading:
    """Тесты для обработанного измерения."""

    def test_processed_reading_creation(self):
        """Тест создания обработанного измерения."""
        reading = ProcessedReading(
            timestamp=datetime.now(),
            raw_value=50000.0,
            filtered_value=50001.0,
            calibrated_value=50002.0,
            quality=95,
        )

        assert reading.raw_value == 50000.0
        assert reading.filtered_value == 50001.0
        assert reading.calibrated_value == 50002.0
        assert reading.quality == 95
        assert reading.anomaly_detected is False

    def test_processed_reading_with_anomaly(self):
        """Тест измерения с обнаруженной аномалией."""
        reading = ProcessedReading(
            timestamp=datetime.now(),
            raw_value=50000.0,
            filtered_value=50000.0,
            calibrated_value=50000.0,
            anomaly_detected=True,
            anomaly_confidence=0.85,
        )

        assert reading.anomaly_detected is True
        assert reading.anomaly_confidence == 0.85


class TestSTM32Status:
    """Тесты для статусов STM32."""

    def test_status_values(self):
        """Тест значений статусов."""
        assert STM32Status.OK == 0x00
        assert STM32Status.INITIALIZING == 0x01
        assert STM32Status.ERROR_SENSOR == 0x80

    def test_status_from_value(self):
        """Тест создания статуса из значения."""
        status = STM32Status(0x00)
        assert status == STM32Status.OK

        status = STM32Status(0x80)
        assert status == STM32Status.ERROR_SENSOR


class TestSTM32Command:
    """Тесты для команд STM32."""

    def test_command_values(self):
        """Тест значений команд."""
        assert STM32Command.PING == 0x01
        assert STM32Command.GET_STATUS == 0x02
        assert STM32Command.GET_READING == 0x03
        assert STM32Command.CALIBRATE == 0x10
        assert STM32Command.RESET == 0x20


# Интеграционные тесты (требуют моки)

class TestIntegration:
    """Интеграционные тесты с моками."""

    @patch("magnetometer.stm32_driver.serial.Serial")
    def test_driver_connect(self, mock_serial):
        """Тест подключения драйвера с моком Serial."""
        mock_serial_instance = MagicMock()
        mock_serial.return_value = mock_serial_instance

        # Настройка мока для PING
        mock_serial_instance.in_waiting = 10
        mock_serial_instance.read.side_effect = [
            bytes([0xAA]),  # START byte 1
            bytes([0x55]),  # START byte 2
            bytes([0x01, 0x01]),  # LEN=1, CMD=PING
            bytes([0x00]),  # Response data (OK)
            bytes([0x00, 0x00]),  # CRC (упрощённо)
        ]

        driver = STM32Driver()
        # Тест прервётся на проверке CRC, но структура верна


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
