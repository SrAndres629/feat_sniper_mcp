import sys
import asyncio
import zmq
import zmq.asyncio
import logging
import json

# --- WINDOWS FIX CRÍTICO ---
# Esto evita el error "Proactor event loop" en Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger("MT5_Bridge.ZMQ")

class ZMQBridge:
    def __init__(self, pub_port=5556, sub_port=5555):
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.context = zmq.asyncio.Context()
        self.pub_socket = None
        self.sub_socket = None
        self.running = False
        self.callbacks = []

    async def start(self, callback=None):
        """Inicia los sockets ZMQ y el loop de escucha."""
        if callback:
            self.add_callback(callback)
            
        try:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind(f"tcp://0.0.0.0:{self.pub_port}")

            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.bind(f"tcp://0.0.0.0:{self.sub_port}")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

            self.running = True
            logger.info(f"ZMQ Bridge LISTENING on tcp://0.0.0.0:{self.sub_port}")
            
            asyncio.create_task(self._listen())
            
        except zmq.ZMQError as e:
            # === RESILIENCE FIX ===
            # Si el puerto está ocupado, no crashear el servidor completo
            # Solo log warning y continuar sin ZMQ (MCP funciona sin él)
            logger.warning(f"ZMQ Bridge DISABLED: {e} - El servidor MCP funcionará sin streaming ZMQ.")
            self.running = False
            # NO re-raise - permitir que el servidor continúe

    async def stop(self):
        """Cierra conexiones limpiamente."""
        self.running = False
        if self.pub_socket: self.pub_socket.close()
        if self.sub_socket: self.sub_socket.close()
        self.context.term()
        logger.info("ZMQ Bridge detenido.")

    def add_callback(self, callback):
        self.callbacks.append(callback)

    async def _listen(self):
        while self.running:
            try:
                msg_string = await self.sub_socket.recv_string()
                # [AUDIT] Structured Traceability - INPUT FLOW
                logger.info("INPUT_FLOW", extra={
                     "module": "ZMQ_BRIDGE", 
                     "action": "RECEIVE", 
                     "size": len(msg_string), 
                     "target": "internal_callback"
                })
                try:
                    data = json.loads(msg_string)
                    for cb in self.callbacks:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(data)
                        else:
                            cb(data)
                except json.JSONDecodeError:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                if self.running:
                    await asyncio.sleep(1)

    async def send_command(self, action: str, **params):
        if not self.running:
            return
        payload = {"action": action, "params": params}
        await self.pub_socket.send_string(json.dumps(payload))

zmq_bridge = ZMQBridge()
