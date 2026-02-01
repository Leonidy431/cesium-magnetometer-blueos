# OceanAI Data Hub

**Универсальная платформа сбора и ИИ-обработки данных океанографического оборудования**

Аппаратно-программный комплекс на базе **Hailo-8L** и **STM32** для сбора, обработки и анализа данных от различного океанографического оборудования с применением искусственного интеллекта.

---

## Обзор системы

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OceanAI Data Hub                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐│
│  │ Магнитометр │  │ Профилограф │  │    Сонар    │  │ Океанографические   ││
│  │   Cs-137    │  │   (SBP)     │  │   (Sonar)   │  │     датчики         ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘│
│         │                │                │                    │           │
│         ▼                ▼                ▼                    ▼           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        STM32H7 MCU Layer                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │   │
│  │  │   UART   │  │   ADC    │  │   SPI    │  │   Digital I/O    │    │   │
│  │  │ 4 канала │  │ 8x 16bit │  │ 2 канала │  │   16 каналов     │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                           │
│                                ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Hailo-8L AI Processor                          │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐    │   │
│  │  │   Anomaly      │  │    Pattern     │  │     Object         │    │   │
│  │  │   Detection    │  │  Recognition   │  │  Classification    │    │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘    │   │
│  └─────────────────────────────┬───────────────────────────────────────┘   │
│                                │                                           │
│                                ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      BlueOS Integration                             │   │
│  │           MAVLink2 REST API  ←→  ROV/AUV Control System             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Поддерживаемое оборудование

### Входные каналы данных

| Оборудование | Интерфейс | Описание |
|--------------|-----------|----------|
| **Цезиевый магнитометр** | UART | Измерение магнитного поля (20-100 μT, 0.001 нТл) |
| **Профилограф (SBP)** | UART/Ethernet | Sub-bottom profiler, профилирование дна |
| **Сонар бокового обзора** | SPI/UART | Side-scan sonar, изображение морского дна |
| **Эхолот** | UART | Батиметрия, измерение глубины |
| **CTD-зонд** | RS-485 | Conductivity, Temperature, Depth |
| **ADCP** | RS-232 | Acoustic Doppler Current Profiler |
| **Турбидиметр** | ADC | Мутность воды |
| **Флуориметр** | ADC | Хлорофилл, флуоресценция |
| **pH/ORP сенсоры** | ADC | Кислотность, окислительно-восстановительный потенциал |
| **Датчики давления** | ADC | Глубина, давление |

### Аналого-цифровые каналы (ADC)

```
STM32H7 ADC Configuration:
├── ADC1: 8 каналов, 16-bit, до 3.6 MSPS
│   ├── CH0-3: Океанографические датчики (pH, турбидность, и т.д.)
│   └── CH4-7: Резервные/пользовательские
├── ADC2: 8 каналов, синхронизация с ADC1
│   └── Дифференциальный режим для прецизионных измерений
└── ADC3: 8 каналов, независимый
    └── Мониторинг питания и температуры
```

### Цифровые каналы (GPIO)

```
Digital I/O Configuration:
├── Входы (16 каналов):
│   ├── DI0-7:  Сигналы от внешнего оборудования
│   └── DI8-15: Триггеры, синхронизация
└── Выходы (16 каналов):
    ├── DO0-7:  Управление реле, клапанами
    └── DO8-15: PWM, управление приводами
```

---

## Аппаратная платформа

### Hailo-8L AI Accelerator

| Параметр | Значение |
|----------|----------|
| Производительность | 13 TOPS |
| Энергопотребление | 2.5W типичное |
| Интерфейс | PCIe Gen3 x4 / M.2 |
| Поддержка фреймворков | TensorFlow, PyTorch, ONNX |
| Latency | < 10ms для инференса |

**ИИ-возможности:**
- Обнаружение магнитных аномалий в реальном времени
- Классификация объектов на изображениях сонара
- Сегментация профилей морского дна
- Фильтрация шумов нейросетью
- Детекция подводных объектов (трубопроводы, кабели, затонувшие объекты)

### STM32H7 Микроконтроллер

| Параметр | Значение |
|----------|----------|
| Ядро | ARM Cortex-M7 @ 480MHz |
| RAM | 1MB internal + external |
| Flash | 2MB |
| ADC | 3x 16-bit, до 3.6 MSPS |
| UART | 8 каналов |
| SPI | 6 каналов |
| I2C | 4 канала |
| Ethernet | 10/100 Mbps |
| USB | HS OTG |

---

## Структура проекта

```
cesium-magnetometer-blueos/
├── magnetometer/                    # Python модуль
│   ├── main.py                      # Точка входа
│   ├── cesium_magnetometer.py       # Драйвер магнитометра
│   ├── stm32_driver.py              # Драйвер STM32
│   ├── hailo_processor.py           # ИИ-обработка Hailo-8L
│   ├── blueos_extension.py          # BlueOS интеграция
│   ├── config.py                    # Конфигурация
│   └── requirements.txt
│
├── firmware/                        # [В разработке] STM32 прошивка
│   ├── Core/
│   │   ├── Src/
│   │   │   ├── main.c
│   │   │   ├── adc_manager.c        # Управление ADC
│   │   │   ├── uart_manager.c       # Управление UART
│   │   │   ├── data_aggregator.c    # Агрегация данных
│   │   │   └── protocol.c           # Протокол обмена
│   │   └── Inc/
│   ├── Drivers/
│   │   ├── magnetometer/
│   │   ├── profiler/
│   │   └── sonar/
│   └── CMakeLists.txt
│
├── models/                          # [В разработке] ИИ-модели
│   ├── anomaly_detection/           # Обнаружение аномалий
│   │   ├── model.onnx
│   │   └── model.hef                # Скомпилировано для Hailo
│   ├── sonar_classification/        # Классификация сонара
│   └── seabed_segmentation/         # Сегментация профилей
│
├── tests/
│   └── test_magnetometer.py
│
├── docs/                            # Документация
├── PROJECT_STAGES.md                # Этапы разработки
├── setup.py
└── README.md
```

---

## Быстрый старт

### Требования

- Python 3.9+
- Hailo Runtime (HailoRT) 4.15+
- BlueOS 1.1+
- STM32 firmware (опционально)

### Установка

```bash
# Клонирование
git clone https://github.com/Leonidy431/cesium-magnetometer-blueos.git
cd cesium-magnetometer-blueos

# Зависимости
pip install -r magnetometer/requirements.txt

# Установка пакета
pip install -e .
```

### Запуск

```bash
# Базовый запуск
python -m magnetometer.main

# С указанием порта магнитометра
python -m magnetometer.main --magnetometer-port /dev/ttyUSB0

# С STM32 для многоканального сбора
python -m magnetometer.main --stm32-port /dev/ttyACM0

# Без Hailo AI (только сбор данных)
python -m magnetometer.main --no-hailo
```

---

## Конфигурация

```json
{
  "magnetometer": {
    "enabled": true,
    "port": "/dev/ttyUSB0",
    "baudrate": 115200,
    "sample_rate": 100
  },

  "stm32": {
    "enabled": true,
    "port": "/dev/ttyACM0",
    "baudrate": 921600
  },

  "adc_channels": {
    "ch0": {"name": "pH", "scale": 14.0, "offset": 0},
    "ch1": {"name": "turbidity", "scale": 1000, "unit": "NTU"},
    "ch2": {"name": "chlorophyll", "scale": 100, "unit": "μg/L"},
    "ch3": {"name": "pressure", "scale": 1000, "unit": "dbar"}
  },

  "digital_inputs": {
    "di0": {"name": "trigger", "edge": "rising"},
    "di1": {"name": "sync_pulse", "edge": "both"}
  },

  "uart_channels": {
    "uart1": {"device": "profiler", "baudrate": 115200},
    "uart2": {"device": "sonar", "baudrate": 460800},
    "uart3": {"device": "ctd", "baudrate": 9600}
  },

  "hailo": {
    "enabled": true,
    "models": {
      "anomaly": "/models/anomaly_detection.hef",
      "sonar": "/models/sonar_classification.hef"
    },
    "batch_size": 32
  },

  "blueos": {
    "api_url": "http://localhost/mavlink2rest",
    "extension_port": 8080
  },

  "logging": {
    "level": "INFO",
    "file": "/var/log/oceanai/data.log",
    "rotation": "daily"
  }
}
```

---

## API

### REST Endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/v1/status` | GET | Статус всех устройств |
| `/v1/magnetometer/data` | GET | Данные магнитометра |
| `/v1/magnetometer/calibrate` | POST | Калибровка |
| `/v1/adc/{channel}` | GET | Данные ADC канала |
| `/v1/adc/all` | GET | Все ADC каналы |
| `/v1/digital/inputs` | GET | Состояние цифровых входов |
| `/v1/digital/outputs` | GET/POST | Управление выходами |
| `/v1/uart/{channel}` | GET | Данные UART канала |
| `/v1/ai/anomalies` | GET | Обнаруженные аномалии |
| `/v1/ai/classification` | GET | Результаты классификации |
| `/v1/recording/start` | POST | Начать запись |
| `/v1/recording/stop` | POST | Остановить запись |

### WebSocket

```javascript
// Подключение к потоку данных
const ws = new WebSocket('ws://localhost:8080/v1/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.magnetometer - данные магнитометра
  // data.adc - данные ADC каналов
  // data.anomalies - обнаруженные аномалии
};
```

---

## Области применения

| Область | Применение |
|---------|------------|
| **Геофизика** | Магнитометрическая съемка, поиск аномалий |
| **Археология** | Поиск затонувших объектов и артефактов |
| **Нефтегаз** | Обследование трубопроводов и кабелей |
| **Гидрография** | Батиметрия, картирование морского дна |
| **Экология** | Мониторинг качества воды |
| **Оборона** | Обнаружение мин и подводных объектов |
| **Научные исследования** | Океанография, морская геология |

---

## Roadmap

- [x] Драйвер цезиевого магнитометра
- [x] Интеграция с BlueOS
- [x] Базовая связь с STM32
- [ ] Прошивка STM32 для многоканального сбора
- [ ] Драйверы профилографа и сонара
- [ ] ИИ-модели для Hailo-8L
- [ ] Web-интерфейс визуализации
- [ ] Запись данных в формате SEG-Y
- [ ] Интеграция с QGIS

---

## Лицензия

MIT License

## Контакты

- GitHub: [@Leonidy431](https://github.com/Leonidy431)
- Проект: [OceanAI Data Hub](https://github.com/Leonidy431/cesium-magnetometer-blueos)
