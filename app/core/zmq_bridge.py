import zmq
import zmq.asyncio
import json
import logging
import asyncio
from typing import Callable, Optional, Any

from app.core.observability import resilient

logger = logging.getLogger("MT5_Bridge.ZMQ")

class ZMQBridge:
    """ZeroMQ ultra-low latency event bridge.
    
    Replaces CSV polling with a high-speed event-driven architecture.
    Handles market data ingestion and internal signal broadcasting.
    
    Attributes:
        host (str): Listening host.
        port (int): Listening port.
        is_running (bool): Lifecycle flag.
        addr (str): Full ZMQ address.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.addr: str = f"tcp://{host}:{port}"
        self.is_running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._callback: Optional[Callable[[dict], Any]] = None

    @resilient(max_retries=5, failure_threshold=3)
    async def start(self, callback: Callable[[dict], Any]) -> None:
        """Starts the ZeroMQ subscriber loop.
        
        Args:
            callback: Async or sync function to handle incoming message dicts.
            
        Raises:
            zmq.ZMQError: If binding fails.
        """
        self._callback = callback
        try:
            self.socket.bind(self.addr)
            self.socket.subscribe("") # Subscribe to all topics
            logger.info(f"ZMQ Bridge LISTENING on {self.addr} (BIND mode)")
        except zmq.ZMQError as e:
            logger.error(f"ZMQ BIND failed: {e}")
            raise
        
        self.is_running = True
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        """Gracefully shuts down the ZeroMQ bridge and releases resources."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.socket.close()
        logger.info("ZMQ Bridge shut down gracefully.")

    async def _listen(self) -> None:
        """Asynchronous listening loop for incoming messages."""
        while self.is_running:
            try:
                # Receive message from MT5
                # Using recv() + decode for UTF-8 robustness
                message_bytes = await self.socket.recv()
                message = message_bytes.decode('utf-8', errors='replace')
                
                if not message:
                    continue

                data = json.loads(message)
                
                if self._callback:
                    if asyncio.iscoroutinefunction(self._callback):
                        await self._callback(data)
                    else:
                        self._callback(data)
                        
            except zmq.ZMQError as e:
                logger.error(f"ZMQ Subscriber Error: {e}")
                await asyncio.sleep(1) # Backoff
            except json.JSONDecodeError as e:
                logger.error(f"ZMQ Payload Decode Error: {e} | Raw: {message[:100]}")
            except Exception as e:
                logger.error(f"Unexpected error in ZMQ loop: {e}", exc_info=True)

# Instancia global para streaming
zmq_bridge = ZMQBridge()
