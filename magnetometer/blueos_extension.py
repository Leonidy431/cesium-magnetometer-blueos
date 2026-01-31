"""
Расширение BlueOS для цезиевого магнитометра.

Модуль интеграции магнитометра с BlueOS — операционной системой
для подводных роботов Blue Robotics. Обеспечивает:
- Регистрацию расширения в BlueOS
- REST API для доступа к данным
- MAVLink интеграцию для телеметрии
- WebSocket для потоковой передачи данных

BlueOS API: https://docs.bluerobotics.com/blueos/
MAVLink: https://mavlink.io/

Author: НПО Лаборатория К
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from aiohttp import web

from .cesium_magnetometer import CesiumMagnetometer, ProcessedReading
from .config import BlueOSConfig, MagnetometerConfig


logger = logging.getLogger(__name__)


class BlueOSExtension:
    """Расширение BlueOS для магнитометра.

    Интегрирует цезиевый магнитометр с экосистемой BlueOS,
    предоставляя веб-интерфейс, REST API и MAVLink телеметрию.

    Attributes:
        config: Конфигурация BlueOS
        magnetometer: Экземпляр магнитометра
        is_running: Флаг состояния сервиса

    Example:
        >>> mag = CesiumMagnetometer()
        >>> extension = BlueOSExtension(mag)
        >>> await extension.start()
        >>> # API доступен на http://localhost:9090
    """

    # Версия расширения
    VERSION = "1.0.0"

    # Порт для веб-сервиса
    WEB_PORT = 9090

    def __init__(
        self,
        magnetometer: CesiumMagnetometer,
        config: Optional[BlueOSConfig] = None,
    ):
        """Инициализация расширения BlueOS.

        Args:
            magnetometer: Экземпляр магнитометра
            config: Конфигурация BlueOS
        """
        self.config = config or BlueOSConfig()
        self.magnetometer = magnetometer

        # Состояние
        self.is_running: bool = False
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        # WebSocket клиенты
        self._ws_clients: List[web.WebSocketResponse] = []
        self._ws_lock = threading.Lock()

        # Буфер последних измерений для API
        self._readings_buffer: List[Dict[str, Any]] = []
        self._buffer_max_size = 1000

        # MAVLink (опционально)
        self._mavlink_connection = None

        logger.debug(
            "BlueOSExtension создан: name=%s, port=%d",
            self.config.extension_name,
            self.WEB_PORT,
        )

    async def start(self) -> bool:
        """Запуск расширения BlueOS.

        Регистрирует расширение в BlueOS, запускает веб-сервер
        и начинает отправку телеметрии.

        Returns:
            bool: True если запуск успешен
        """
        if self.is_running:
            logger.warning("Расширение уже запущено")
            return True

        try:
            # Регистрация в BlueOS
            await self._register_extension()

            # Создание веб-приложения
            self._app = web.Application()
            self._setup_routes()

            # Запуск веб-сервера
            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(
                self._runner,
                "0.0.0.0",
                self.WEB_PORT,
            )
            await self._site.start()

            # Подписка на данные магнитометра
            self.magnetometer.start_measurement(
                callback=self._on_magnetometer_reading,
            )

            self.is_running = True
            logger.info(
                "BlueOS расширение запущено на порту %d",
                self.WEB_PORT,
            )
            return True

        except Exception as e:
            logger.error("Ошибка запуска расширения: %s", e)
            return False

    async def stop(self) -> None:
        """Остановка расширения BlueOS."""
        if not self.is_running:
            return

        # Остановка измерений
        self.magnetometer.stop_measurement()

        # Закрытие WebSocket соединений
        with self._ws_lock:
            for ws in self._ws_clients:
                await ws.close()
            self._ws_clients.clear()

        # Остановка веб-сервера
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

        # Отмена регистрации в BlueOS
        await self._unregister_extension()

        self.is_running = False
        logger.info("BlueOS расширение остановлено")

    def _setup_routes(self) -> None:
        """Настройка маршрутов веб-приложения."""
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/v1/status", self._handle_status)
        self._app.router.add_get("/v1/reading", self._handle_reading)
        self._app.router.add_get("/v1/readings", self._handle_readings)
        self._app.router.add_get("/v1/statistics", self._handle_statistics)
        self._app.router.add_post("/v1/calibrate", self._handle_calibrate)
        self._app.router.add_post("/v1/config", self._handle_config)
        self._app.router.add_get("/v1/ws", self._handle_websocket)
        self._app.router.add_get("/register_service", self._handle_register)

    async def _handle_index(self, request: web.Request) -> web.Response:
        """Главная страница с информацией о расширении.

        Args:
            request: HTTP запрос

        Returns:
            web.Response: HTML страница
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cesium Magnetometer - BlueOS Extension</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2196F3; }}
                .status {{ padding: 10px; background: #e8f5e9; border-radius: 5px; }}
                .api {{ margin-top: 20px; }}
                .endpoint {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>Cesium Magnetometer</h1>
            <div class="status">
                <strong>Version:</strong> {self.VERSION}<br>
                <strong>Status:</strong> {'Running' if self.is_running else 'Stopped'}<br>
                <strong>Connected:</strong> {self.magnetometer.is_connected}<br>
            </div>
            <div class="api">
                <h2>API Endpoints</h2>
                <div class="endpoint">GET /v1/status - Device status</div>
                <div class="endpoint">GET /v1/reading - Current reading</div>
                <div class="endpoint">GET /v1/readings - Buffered readings</div>
                <div class="endpoint">GET /v1/statistics - Measurement statistics</div>
                <div class="endpoint">POST /v1/calibrate - Start calibration</div>
                <div class="endpoint">POST /v1/config - Update configuration</div>
                <div class="endpoint">GET /v1/ws - WebSocket stream</div>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type="text/html")

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Обработчик запроса статуса.

        Args:
            request: HTTP запрос

        Returns:
            web.Response: JSON со статусом
        """
        status = self.magnetometer.get_status()
        status["extension_version"] = self.VERSION
        status["blueos_registered"] = self.is_running

        return web.json_response(status)

    async def _handle_reading(self, request: web.Request) -> web.Response:
        """Обработчик запроса текущего измерения.

        Args:
            request: HTTP запрос

        Returns:
            web.Response: JSON с измерением
        """
        try:
            reading = self.magnetometer.get_single_reading()
            return web.json_response(self._reading_to_dict(reading))
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500,
            )

    async def _handle_readings(self, request: web.Request) -> web.Response:
        """Обработчик запроса буферизованных измерений.

        Args:
            request: HTTP запрос

        Returns:
            web.Response: JSON со списком измерений
        """
        # Параметр limit
        limit = int(request.query.get("limit", 100))
        limit = min(limit, 1000)

        readings = self._readings_buffer[-limit:]
        return web.json_response({"readings": readings, "count": len(readings)})

    async def _handle_statistics(self, request: web.Request) -> web.Response:
        """Обработчик запроса статистики.

        Args:
            request: HTTP запрос

        Returns:
            web.Response: JSON со статистикой
        """
        stats = self.magnetometer.get_statistics()
        return web.json_response(asdict(stats))

    async def _handle_calibrate(self, request: web.Request) -> web.Response:
        """Обработчик запроса калибровки.

        Args:
            request: HTTP запрос

        Returns:
            web.Response: JSON с результатом
        """
        try:
            # Калибровка в отдельном потоке (блокирующая операция)
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                self.magnetometer.calibrate,
            )

            return web.json_response({
                "success": success,
                "message": "Calibration complete" if success else "Calibration failed",
            })
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    async def _handle_config(self, request: web.Request) -> web.Response:
        """Обработчик обновления конфигурации.

        Args:
            request: HTTP запрос

        Returns:
            web.Response: JSON с результатом
        """
        try:
            data = await request.json()

            # Обновление разрешённых параметров
            if "sample_rate" in data:
                self.magnetometer.config.sample_rate = int(data["sample_rate"])

            if "filter_type" in data:
                from .config import FilterType
                self.magnetometer.set_filter(
                    FilterType[data["filter_type"]],
                )

            return web.json_response({
                "success": True,
                "config": self.magnetometer.config.to_dict(),
            })
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=400,
            )

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Обработчик WebSocket соединения.

        Устанавливает WebSocket соединение для потоковой
        передачи данных магнитометра в реальном времени.

        Args:
            request: HTTP запрос

        Returns:
            web.WebSocketResponse: WebSocket соединение
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        with self._ws_lock:
            self._ws_clients.append(ws)

        logger.info("WebSocket клиент подключён")

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Обработка команд от клиента
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_command(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_json({"error": "Invalid JSON"})

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        "WebSocket ошибка: %s",
                        ws.exception(),
                    )
        finally:
            with self._ws_lock:
                if ws in self._ws_clients:
                    self._ws_clients.remove(ws)

            logger.info("WebSocket клиент отключён")

        return ws

    async def _handle_ws_command(
        self,
        ws: web.WebSocketResponse,
        data: dict,
    ) -> None:
        """Обработка команды WebSocket.

        Args:
            ws: WebSocket соединение
            data: Данные команды
        """
        command = data.get("command")

        if command == "ping":
            await ws.send_json({"response": "pong"})

        elif command == "get_status":
            status = self.magnetometer.get_status()
            await ws.send_json({"type": "status", "data": status})

        elif command == "subscribe":
            # Подписка уже активна по умолчанию
            await ws.send_json({"response": "subscribed"})

        else:
            await ws.send_json({"error": f"Unknown command: {command}"})

    async def _handle_register(self, request: web.Request) -> web.Response:
        """Обработчик для BlueOS service discovery.

        Args:
            request: HTTP запрос

        Returns:
            web.Response: JSON с информацией о сервисе
        """
        return web.json_response({
            "name": self.config.extension_name,
            "description": "Cesium Magnetometer for underwater surveys",
            "icon": "mdi-magnet",
            "company": "NPO Laboratory K",
            "version": self.VERSION,
            "webpage": f"http://localhost:{self.WEB_PORT}",
            "api": f"http://localhost:{self.WEB_PORT}/v1",
        })

    async def _register_extension(self) -> None:
        """Регистрация расширения в BlueOS."""
        blueos_url = (
            f"http://{self.config.host}:{self.config.port}"
            f"/helper/v1.0/services"
        )

        service_data = {
            "name": self.config.extension_name,
            "port": self.WEB_PORT,
            "type": "sensor",
            "metadata": {
                "description": "Cesium Magnetometer",
                "version": self.VERSION,
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    blueos_url,
                    json=service_data,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        logger.info("Расширение зарегистрировано в BlueOS")
                    else:
                        logger.warning(
                            "Не удалось зарегистрировать в BlueOS: %d",
                            resp.status,
                        )
        except Exception as e:
            logger.warning("BlueOS недоступен: %s", e)

    async def _unregister_extension(self) -> None:
        """Отмена регистрации расширения в BlueOS."""
        blueos_url = (
            f"http://{self.config.host}:{self.config.port}"
            f"/helper/v1.0/services/{self.config.extension_name}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    blueos_url,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        logger.info("Регистрация в BlueOS отменена")
        except Exception:
            pass  # Игнорируем ошибки при отмене регистрации

    def _on_magnetometer_reading(self, reading: ProcessedReading) -> None:
        """Callback для новых измерений магнитометра.

        Args:
            reading: Обработанное измерение
        """
        reading_dict = self._reading_to_dict(reading)

        # Добавление в буфер
        self._readings_buffer.append(reading_dict)
        if len(self._readings_buffer) > self._buffer_max_size:
            self._readings_buffer = self._readings_buffer[-self._buffer_max_size // 2:]

        # Отправка по WebSocket (асинхронно)
        asyncio.run_coroutine_threadsafe(
            self._broadcast_reading(reading_dict),
            asyncio.get_event_loop(),
        )

    async def _broadcast_reading(self, reading_dict: dict) -> None:
        """Рассылка измерения по WebSocket.

        Args:
            reading_dict: Измерение в виде словаря
        """
        message = {
            "type": "reading",
            "data": reading_dict,
        }

        with self._ws_lock:
            dead_clients = []

            for ws in self._ws_clients:
                try:
                    if not ws.closed:
                        await ws.send_json(message)
                    else:
                        dead_clients.append(ws)
                except Exception:
                    dead_clients.append(ws)

            # Удаление отключённых клиентов
            for ws in dead_clients:
                self._ws_clients.remove(ws)

    @staticmethod
    def _reading_to_dict(reading: ProcessedReading) -> dict:
        """Преобразование измерения в словарь.

        Args:
            reading: Измерение

        Returns:
            dict: Словарь с данными
        """
        return {
            "timestamp": reading.timestamp.isoformat(),
            "raw_value_nT": reading.raw_value,
            "filtered_value_nT": reading.filtered_value,
            "calibrated_value_nT": reading.calibrated_value,
            "gradient_nT_m": reading.gradient,
            "latitude": reading.latitude,
            "longitude": reading.longitude,
            "depth_m": reading.depth,
            "quality_percent": reading.quality,
            "anomaly_detected": reading.anomaly_detected,
            "anomaly_confidence": reading.anomaly_confidence,
        }


async def run_extension(magnetometer: CesiumMagnetometer) -> None:
    """Запуск расширения BlueOS как основного процесса.

    Args:
        magnetometer: Экземпляр магнитометра

    Example:
        >>> import asyncio
        >>> from magnetometer import CesiumMagnetometer
        >>> from magnetometer.blueos_extension import run_extension
        >>>
        >>> mag = CesiumMagnetometer()
        >>> mag.connect()
        >>> asyncio.run(run_extension(mag))
    """
    extension = BlueOSExtension(magnetometer)
    await extension.start()

    try:
        # Бесконечный цикл
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await extension.stop()
