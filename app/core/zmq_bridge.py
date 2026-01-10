import zmq
import zmq.asyncio
import json
import logging
import asyncio
from typing import Callable, Optional

logger = logging.getLogger("MT5_Bridge.ZMQ")

class ZMQBridge:
    """
    Puente de ultra-baja latencia usando ZeroMQ.
    Sustituye el polling de archivos CSV por una arquitectura Event-Driven.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.addr = f"tcp://{host}:{port}"
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._callback: Optional[Callable] = None

    async def start(self, callback: Callable):
        """Inicia el suscriptor ZeroMQ."""
        self._callback = callback
        try:
            self.socket.bind(self.addr)
            self.socket.subscribe("") # Suscribirse a todos los mensajes
            logger.info(f"ZMQ Bridge ESCUCHANDO en {self.addr} (BIND mode)")
        except Exception as e:
            logger.error(f"Error haciendo BIND en ZMQ: {e}")
            raise e
        
        self.is_running = True
        self._task = asyncio.create_task(self._listen())

    async def stop(self):
        """Detiene el bridge."""
        self.is_running = False
        if self._task:
            self._task.cancel()
        self.socket.close()
        logger.info("ZMQ Bridge detenido.")

    async def _listen(self):
        """Loop de escucha asíncrono."""
        while self.is_running:
            try:
                # Recibir mensaje de MT5
                message = await self.socket.recv_string()
                print(f"RAW ZMQ RECIBIDO: {message[:100]}...", flush=True)
                data = json.loads(message)
                
                if self._callback:
                    if asyncio.iscoroutinefunction(self._callback):
                        await self._callback(data)
                    else:
                        self._callback(data)
                        
            except zmq.ZMQError as e:
                logger.error(f"Error en ZMQ Subscriber: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error procesando mensaje ZMQ: {e}")

# Instancia global para streaming
zmq_bridge = ZMQBridge()
