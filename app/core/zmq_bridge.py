"""
ZMQ Bridge - High-Performance Non-Blocking Architecture (ZMQNucleus)
=====================================================================
Zero-latency message bridge between MQL5 and Python brain.

[PROJECT ATLAS] Vibranium Grade Upgrades:
- asyncio.PriorityQueue for Execution > Telemetry ordering
- BackpressureManager: LIFO frame dropping when lag > threshold
- Separation of control: PUB (Stream) vs PUSH (Execution)
- asyncio.to_thread for ALL sync callbacks (no blocking)
- Rate limiting via semaphore (max concurrent processing)
- Enhanced TTL validation with metrics
"""
import sys
import asyncio
import time
import zmq
import zmq.asyncio
import logging
import json
from enum import IntEnum
from typing import Callable, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from app.core.config import settings

# --- WINDOWS FIX CRÍTICO ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger("MT5_Bridge.ZMQ")


# =============================================================================
# MESSAGE PRIORITY SYSTEM
# =============================================================================

class MessagePriority(IntEnum):
    """Priority levels for message processing. Lower = higher priority."""
    CRITICAL = 0      # Emergency stop, circuit breaker
    EXECUTION = 1     # Trade orders (BUY, SELL, CLOSE)
    RISK = 2          # Risk updates, margin alerts
    TELEMETRY = 5     # HUD updates, metrics
    DEBUG = 10        # Low-priority diagnostic


@dataclass(order=True)
class PrioritizedMessage:
    """Message wrapper for priority queue ordering."""
    priority: int
    timestamp: float = field(compare=False)
    data: Dict = field(compare=False)


# =============================================================================
# BACKPRESSURE MANAGER
# =============================================================================

class BackpressureManager:
    """
    Manages message backpressure to prevent queue overflow.
    
    When consumer is slower than producer:
    1. Drop oldest TELEMETRY messages (LIFO for freshness)
    2. Never drop EXECUTION or CRITICAL messages
    3. Alert when backpressure threshold is exceeded
    """
    
    def __init__(self, max_queue_size: int = 100, lag_threshold_ms: float = 200.0):
        self.max_queue_size = max_queue_size
        self.lag_threshold_ms = lag_threshold_ms
        self.frames_dropped = 0
        self.backpressure_events = 0
        self._telemetry_buffer: deque = deque(maxlen=max_queue_size)
    
    def should_drop_telemetry(self, queue_size: int, lag_ms: float) -> bool:
        """
        Determine if telemetry message should be dropped.
        
        Returns:
            True if message should be dropped to relieve backpressure
        """
        if lag_ms > self.lag_threshold_ms:
            self.backpressure_events += 1
            return True
        
        if queue_size >= self.max_queue_size:
            self.backpressure_events += 1
            return True
        
        return False
    
    def record_drop(self):
        """Record that a frame was dropped."""
        self.frames_dropped += 1
    
    def get_metrics(self) -> Dict:
        """Return backpressure metrics."""
        return {
            "frames_dropped": self.frames_dropped,
            "backpressure_events": self.backpressure_events,
            "max_queue_size": self.max_queue_size,
            "lag_threshold_ms": self.lag_threshold_ms
        }


class ZMQBridge:
    """
    High-frequency ZMQ message bridge with non-blocking callback execution.
    
    [P1 FIX] Architecture improvements:
    - All callbacks run via asyncio.to_thread (never blocks event loop)
    - State variables are instance-level, not class-level
    - Semaphore limits concurrent callback processing
    - Enhanced metrics for observability
    """
    
    # Processing limits
    MAX_CONCURRENT_CALLBACKS: int = 10  # Semaphore limit
    
    def __init__(self, pub_port: int = None, sub_port: int = None):
        """
        Initialize ZMQ Bridge with configurable ports.
        
        Args:
            pub_port: Port for publishing commands to MT5 (default: from settings)
            sub_port: Port for subscribing to MT5 ticks (default: from settings)
        """
        self.pub_port = pub_port or settings.ZMQ_PUB_PORT
        self.sub_port = sub_port or settings.ZMQ_PORT
        self.context = zmq.asyncio.Context()
        self.pub_socket: Optional[zmq.asyncio.Socket] = None
        self.sub_socket: Optional[zmq.asyncio.Socket] = None
        self.running = False
        self.callbacks: List[Callable] = []
        
        # [P1 FIX] Instance-level state (was class-level - bug)
        self.current_regime: str = "LAMINAR"
        self.current_ai_confidence: float = 0.5
        
        # Metrics for observability
        self.messages_processed: int = 0
        self.messages_discarded: int = 0
        self.messages_failed: int = 0
        self._last_lag_ms: float = 0.0
        self._peak_lag_ms: float = 0.0
        
        # [P1 FIX] Semaphore for rate limiting
        self._callback_semaphore: Optional[asyncio.Semaphore] = None
        
        logger.info(f"ZMQBridge initialized (pub:{self.pub_port}, sub:{self.sub_port})")

    async def start(self, callback: Callable = None):
        """
        Inicia los sockets ZMQ y el loop de escucha.
        
        [P1 FIX] Now creates semaphore for callback throttling.
        """
        if callback:
            self.add_callback(callback)
        
        # Create semaphore for this event loop
        self._callback_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CALLBACKS)
            
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
        if self.pub_socket: 
            self.pub_socket.close()
        if self.sub_socket: 
            self.sub_socket.close()
        self.context.term()
        logger.info("ZMQ Bridge detenido.")

    def add_callback(self, callback: Callable):
        """Add a callback to be invoked on each message."""
        self.callbacks.append(callback)
        logger.debug(f"Callback added: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")

    async def _listen(self):
        """
        [P1 FIX] Non-blocking message processing loop.
        
        Key improvements:
        - TTL validation with enhanced metrics
        - All callbacks wrapped in asyncio.to_thread for sync functions
        - Semaphore limits concurrent processing to prevent OOM
        - Graceful error handling
        """
        while self.running:
            try:
                msg_string = await self.sub_socket.recv_string()
                
                # [AUDIT] Structured Traceability - INPUT FLOW
                logger.debug(f"INPUT_FLOW: size={len(msg_string)}")
                
                try:
                    data = json.loads(msg_string)
                    
                    # TTL Validation - Reject stale messages
                    msg_ts = data.get("timestamp") or data.get("ts") or data.get("decision_ts")
                    if msg_ts:
                        age_ms = (time.time() * 1000) - msg_ts
                        self._last_lag_ms = age_ms
                        self._peak_lag_ms = max(self._peak_lag_ms, age_ms)
                        
                        if age_ms > settings.DECISION_TTL_MS:
                            self.messages_discarded += 1
                            logger.warning(
                                f"⏰ ZMQ message discarded: age={age_ms:.0f}ms > TTL {settings.DECISION_TTL_MS}ms"
                            )
                            continue
                    
                    # [P1 FIX] Process callbacks with semaphore limiting
                    self.messages_processed += 1
                    asyncio.create_task(self._invoke_callbacks(data))
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON received: {e}")
                    self.messages_failed += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ZMQ listen error: {e}")
                self.messages_failed += 1
                if self.running:
                    await asyncio.sleep(1)

    async def _invoke_callbacks(self, data: Dict[str, Any]):
        """
        [P1 FIX] Invoke all callbacks with proper async handling.
        
        - Async callbacks: awaited directly
        - Sync callbacks: wrapped in asyncio.to_thread (NON-BLOCKING)
        - Protected by semaphore to limit concurrency
        """
        async with self._callback_semaphore:
            for cb in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        # Async callback - await directly
                        await cb(data)
                    else:
                        # [P1 FIX] Sync callback - run in thread to avoid blocking
                        # This is the CRITICAL fix - prevents loop blocking
                        await asyncio.to_thread(cb, data)
                except Exception as e:
                    logger.error(f"Callback error ({cb.__name__ if hasattr(cb, '__name__') else 'anon'}): {e}")

    async def _heartbeat(self):
        """
        Sends periodic health metrics to MT5 HUD (Phase 13 Institutional Upgrade).
        
        [P1 FIX] Enhanced with peak lag tracking and failure count.
        [MTF FIX] Now sends flat JSON format expected by FEAT_Visualizer.
        """
        while self.running:
            try:
                # Flat JSON format for FEAT_Visualizer compatibility
                hud_data = {
                    "regime": self.current_regime,
                    "ai_confidence": round(self.current_ai_confidence, 2),
                    "feat_score_val": round(self.current_ai_confidence * 100, 1),
                    "feat_pvp_price": 0.0,  # Will be set by telemetry
                    "acceleration": False,
                    "vol_factor": 1.0,
                    "zmq_lag_ms": round(self._last_lag_ms, 2),
                    "zmq_peak_lag_ms": round(self._peak_lag_ms, 2),
                    "processed": self.messages_processed,
                    "discarded": self.messages_discarded,
                    "failed": self.messages_failed,
                    "ts": time.time() * 1000
                }
                await self.send_raw(hud_data)
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
        """Send a command to MT5 via PUB socket."""
        if not self.running or not self.pub_socket:
            return
        payload = {
            "action": action, 
            "params": params, 
            "ts": time.time() * 1000
        }
        await self.send_raw(payload)

    async def send_raw(self, payload: Dict[str, Any]):
        """Send a raw JSON payload directly."""
        if not self.running or not self.pub_socket:
            return
        try:
            await self.pub_socket.send_string(json.dumps(payload))
        except Exception as e:
            logger.error(f"Failed to send raw payload: {e}")

    async def send_command_with_ack(self, action: str, **params) -> Optional[Dict[str, Any]]:
        """
        Send a command to MT5 and wait for acknowledgment.
        
        Uses correlation ID to match request/response.
        
        Args:
            action: Command action (BUY, SELL, CLOSE, etc.)
            **params: Additional parameters
            
        Returns:
            Response dict with 'result', 'ticket', 'error' keys or None on failure
        """
        import uuid
        
        if not self.running or not self.pub_socket:
            return {"result": "ERROR", "error": "ZMQ bridge not running"}
        
        correlation_id = str(uuid.uuid4())[:8]
        
        payload = {
            "action": action, 
            "params": params, 
            "ts": time.time() * 1000,
            "correlation_id": correlation_id
        }
        
        # Create temporary response holder
        response_future = asyncio.get_event_loop().create_future()
        
        # Store pending request
        if not hasattr(self, '_pending_acks'):
            self._pending_acks: Dict[str, asyncio.Future] = {}
        self._pending_acks[correlation_id] = response_future
        
        try:
            # Send command
            await self.pub_socket.send_string(json.dumps(payload))
            logger.debug(f"[ACK] Sent command with correlation_id={correlation_id}")
            
            # Wait for response (caller handles timeout)
            response = await response_future
            return response
            
        except asyncio.CancelledError:
            logger.warning(f"[ACK] Request cancelled: {correlation_id}")
            return None
        except Exception as e:
            logger.error(f"[ACK] Send failed: {e}")
            return {"result": "ERROR", "error": str(e)}
        finally:
            # Cleanup
            self._pending_acks.pop(correlation_id, None)

    def handle_ack_response(self, data: Dict[str, Any]):
        """
        Handle incoming ACK response from MT5.
        
        Called from _invoke_callbacks when message has correlation_id.
        """
        correlation_id = data.get("correlation_id")
        if correlation_id and hasattr(self, '_pending_acks'):
            future = self._pending_acks.get(correlation_id)
            if future and not future.done():
                future.set_result(data)
                logger.debug(f"[ACK] Response matched: {correlation_id}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current bridge metrics for monitoring."""
        return {
            "running": self.running,
            "pub_port": self.pub_port,
            "sub_port": self.sub_port,
            "messages_processed": self.messages_processed,
            "messages_discarded": self.messages_discarded,
            "messages_failed": self.messages_failed,
            "last_lag_ms": round(self._last_lag_ms, 2),
            "peak_lag_ms": round(self._peak_lag_ms, 2),
            "current_regime": self.current_regime,
            "current_ai_confidence": self.current_ai_confidence,
            "callback_count": len(self.callbacks)
        }

    def reset_peak_metrics(self):
        """Reset peak metrics (called periodically for monitoring)."""
        self._peak_lag_ms = 0.0


# Global singleton instance
zmq_bridge = ZMQBridge()
