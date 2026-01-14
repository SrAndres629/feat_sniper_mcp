import sys
import asyncio
import time
import zmq
import zmq.asyncio
import logging
import json

from app.core.config import settings

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
        # Metrics for observability
        self.messages_processed = 0
        self.messages_discarded = 0
        self._last_lag_ms = 0.0

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
            asyncio.create_task(self._heartbeat())
            
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
                    
                    # TTL Validation - Reject stale messages
                    msg_ts = data.get("timestamp") or data.get("ts") or data.get("decision_ts")
                    if msg_ts:
                        age_ms = (time.time() * 1000) - msg_ts
                        self._last_lag_ms = age_ms
                        if age_ms > settings.DECISION_TTL_MS:
                            self.messages_discarded += 1
                            logger.warning(f"⏰ ZMQ message discarded: age={age_ms:.0f}ms > TTL {settings.DECISION_TTL_MS}ms")
                            continue
                    
                    self.messages_processed += 1
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
    # Phase 13: HUD State
    current_regime = "LAMINAR"
    current_ai_confidence = 0.5
    
    async def _heartbeat(self):
        """Sends periodic health metrics to MT5 HUD (Phase 13 Institutional Upgrade)."""
        while self.running:
            try:
                await self.send_command(
                    "HUD_UPDATE",
                    regime=self.current_regime,
                    ai_confidence=round(self.current_ai_confidence * 100, 1),
                    zmq_lag_ms=round(self._last_lag_ms, 2),
                    processed=self.messages_processed,
                    discarded=self.messages_discarded
                )
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"HUD Update failed: {e}")
                await asyncio.sleep(5)

    def update_hud_state(self, regime: str = None, ai_confidence: float = None):
        """Updates HUD telemetry state for next broadcast."""
        if regime:
            self.current_regime = regime
        if ai_confidence is not None:
            self.current_ai_confidence = ai_confidence

    async def send_command(self, action: str, **params):
        if not self.running:
            return
        payload = {"action": action, "params": params, "ts": time.time() * 1000}
        await self.pub_socket.send_string(json.dumps(payload))

zmq_bridge = ZMQBridge()
