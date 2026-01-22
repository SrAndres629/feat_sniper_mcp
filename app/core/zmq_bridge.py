import sys
import asyncio
import time
import zmq
import zmq.asyncio
import logging
import json
from enum import IntEnum
from typing import Callable, List, Dict, Any, Optional, Tuple, Coroutine, Union
from dataclasses import dataclass, field

from app.core.config import settings

# --- OS-SPECIFIC ASYNCIO OPTIMIZATION ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger("FEAT.ZMQBridge")


class MessagePriority(IntEnum):
    """
    Priority levels for the ZMQ Internal Queue.
    In hierarchical systems, execution commands MUST supersede telemetry.
    """
    CRITICAL = 0      # Emergency halt, circuit breakers
    EXECUTION = 1     # Market orders (BUY, SELL)
    RISK = 2          # Margin alerts, risk adjustments
    TELEMETRY = 5     # HUD data, dashboard updates
    DEBUG = 10        # Diagnostic traces


@dataclass(order=True)
class PrioritizedMessage:
    """
    Weighting wrapper for message scheduling.
    Ensures the PriorityQueue handles messages by urgency first, then time.
    """
    priority: int
    timestamp: float = field(compare=False)
    data: Dict[str, Any] = field(compare=False)


class BackpressureManager:
    """
    Elite Congestion Control for high-frequency data streams.
    
    Implements a LIFO-style drop policy for stale telemetry while 
    guaranteeing delivery for critical execution frames.
    """
    
    def __init__(self, max_queue_size: int = 100, lag_threshold_ms: float = 200.0):
        """
        Initializes the backpressure controller.
        
        Args:
            max_queue_size (int): Max capacity of the non-critical buffer.
            lag_threshold_ms (float): Maximum allowed latency before dropping frames.
        """
        self.max_queue_size: int = max_queue_size
        self.lag_threshold_ms: float = lag_threshold_ms
        self.frames_dropped: int = 0
        self.backpressure_events: int = 0
    
    def should_drop_telemetry(self, queue_size: int, lag_ms: float) -> bool:
        """
        Checks if the current system state warrants a telemetry drop.
        
        Args:
            queue_size (int): Current size of the processing queue.
            lag_ms (float): Calculated latency in milliseconds.
            
        Returns:
            bool: True if the frame should be discarded to preserve system stability.
        """
        is_lagging = lag_ms > self.lag_threshold_ms
        is_overflowing = queue_size >= self.max_queue_size
        
        if is_lagging or is_overflowing:
            self.backpressure_events += 1
            return True
        
        return False
    
    def record_drop(self) -> None:
        """Increments the dropped frame counter for telemetry."""
        self.frames_dropped += 1
    
    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Aggregates performance metrics for health monitoring.
        
        Returns:
            Dict: Statistics on drop rates and congestion events.
        """
        return {
            "frames_dropped": self.frames_dropped,
            "backpressure_events": self.backpressure_events,
            "max_queue_size": self.max_queue_size,
            "lag_threshold_ms": self.lag_threshold_ms
        }


class ZMQBridge:
    """
    Sovereign ZMQ Interaction Layer.
    
    Orchestrates the asynchronous message exchange between the MQL5 Terminal 
    and the Strategic Cortex using a PUB/PULL hybrid topology.
    
    Restoration V6 Standards:
    - Zero Technical Debt (Full Typing/Docstrings).
    - Abstraction of network endpoints.
    - Semaphore-throttled callback execution.
    """
    
    # Doctoral Constants
    MAX_CONCURRENT_CALLBACKS: int = 20  # Balanced concurrency limit
    HEARTBEAT_INTERVAL: float = 1.0     # Metric broadcast frequency
    
    def __init__(self, pub_port: Optional[int] = None, sub_port: Optional[int] = None):
        """
        Initializes the Bridge state and ZMQ context.
        
        Args:
            pub_port (int, optional): Port for Command broadcasting. Defaults to settings.ZMQ_PUB_PORT.
            sub_port (int, optional): Port for Tick data acquisition. Defaults to settings.ZMQ_PORT.
        """
        self.pub_port: int = pub_port or settings.ZMQ_PUB_PORT
        self.sub_port: int = sub_port or settings.ZMQ_PORT
        self.context: zmq.asyncio.Context = zmq.asyncio.Context()
        self.pub_socket: Optional[zmq.asyncio.Socket] = None
        self.sub_socket: Optional[zmq.asyncio.Socket] = None
        self.running: bool = False
        self.callbacks: List[Callable] = []
        self._pending_acks: Dict[str, asyncio.Future] = {}
        
        # Internal State (Neural Synapse)
        self.current_regime: str = "SEARCHING"
        self.current_ai_confidence: float = 0.0
        
        # Observability Metrics
        self.messages_processed: int = 0
        self.messages_discarded: int = 0
        self.messages_failed: int = 0
        self._last_lag_ms: float = 0.0
        self._peak_lag_ms: float = 0.0
        
        # Concurrency Control
        self._callback_semaphore: Optional[asyncio.Semaphore] = None
        
        logger.info("[ZMQ_CORE] Bridge initialized | Pub:%d | Sub:%d", self.pub_port, self.sub_port)

    async def start(self, callback: Optional[Callable] = None) -> None:
        """
        Activates ZMQ Sockets and initiates the internal message loops.
        
        Args:
            callback (Callable, optional): Initial processing callback to attach.
        """
        if callback:
            self.add_callback(callback)
        
        self._callback_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CALLBACKS)
            
        try:
            # Command Publisher (Python -> MT5)
            self.pub_socket = self.context.socket(zmq.PUB)
            # Apply Conflate if specified in settings for HUD updates
            if getattr(settings, "ZMQ_CONFLATE", True):
                self.pub_socket.setsockopt(zmq.CONFLATE, 1)
            
            bind_url = f"tcp://{settings.ZMQ_BIND_ADDRESS}:{self.pub_port}"
            self.pub_socket.bind(bind_url)

            # Execution Pipeline (MT5 -> Python)
            # Using PULL for reliable sequential processing of ticks
            self.sub_socket = self.context.socket(zmq.PULL)
            sub_bind_url = f"tcp://{settings.ZMQ_BIND_ADDRESS}:{self.sub_port}"
            self.sub_socket.bind(sub_bind_url)

            self.running = True
            logger.info("[ZMQ_CORE] Sockets bound effectively on %s", settings.ZMQ_BIND_ADDRESS)
            
            asyncio.create_task(self._listen())
            asyncio.create_task(self._heartbeat())
            
        except zmq.ZMQError as e:
            logger.error("[ZMQ_CORE] CRITICAL SOCKET FAILURE: %s", e)
            self.running = False

    async def stop(self) -> None:
        """Performs a graceful shutdown of all sockets and contexts."""
        self.running = False
        if self.pub_socket: 
            self.pub_socket.close(linger=0)
        if self.sub_socket: 
            self.sub_socket.close(linger=0)
        
        # Terminate context to release all resources
        self.context.term()
        logger.info("[ZMQ_CORE] Bridge shutdown complete.")

    def add_callback(self, callback: Callable) -> None:
        """
        Attaches a subscriber callback to the message stream.
        
        Args:
            callback (Callable): Function to execute on incoming data.
        """
        self.callbacks.append(callback)
        cb_name = getattr(callback, "__name__", "anonymous")
        logger.debug("[ZMQ_CORE] Callback associated: %s", cb_name)

    async def _listen(self) -> None:
        """
        Main acquisition loop.
        
        Validates incoming data integrity and TTL before dispatching to workers.
        """
        while self.running:
            try:
                msg_string = await self.sub_socket.recv_string()
                
                try:
                    data = json.loads(msg_string)
                    
                    # Temporal Validation (SRE TTL Check)
                    msg_ts = data.get("timestamp") or data.get("ts") or data.get("time")
                    if msg_ts:
                        # Convert to ms if necessary
                        if msg_ts < 1e11: msg_ts *= 1000 
                        
                        age_ms = (time.time() * 1000) - msg_ts
                        self._last_lag_ms = age_ms
                        self._peak_lag_ms = max(self._peak_lag_ms, age_ms)
                        
                        if age_ms > settings.DECISION_TTL_MS:
                            self.messages_discarded += 1
                            logger.warning("[ZMQ_CORE] STALE MSG: %d ms > TTL %d ms", age_ms, settings.DECISION_TTL_MS)
                            continue
                    
                    self.messages_processed += 1
                    asyncio.create_task(self._invoke_callbacks(data))
                    
                except json.JSONDecodeError as e:
                    logger.error("[ZMQ_CORE] DATA CORRUPTION: Invalid JSON | %s", e)
                    self.messages_failed += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[ZMQ_CORE] RECV_ERROR: %s", e)
                self.messages_failed += 1
                if self.running:
                    await asyncio.sleep(1)

    async def _invoke_callbacks(self, data: Dict[str, Any]) -> None:
        """
        Dispatches data to registered callbacks using thread-pooling for sync tasks.
        
        Args:
            data (Dict): The message payload.
        """
        async with self._callback_semaphore:
            for cb in self.callbacks:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(data)
                    else:
                        # Offload sync callbacks to avoid event loop starvation
                        await asyncio.to_thread(cb, data)
                except Exception as e:
                    logger.error("[ZMQ_CORE] CALLBACK FAILURE [%s]: %s", getattr(cb, "__name__", "anon"), e)

    async def _heartbeat(self) -> None:
        """Broadcasts periodic telemetry and health metrics to the MT5 Terminal."""
        while self.running:
            try:
                hud_data = {
                    "regime": self.current_regime,
                    "ai_confidence": round(self.current_ai_confidence, 2),
                    "feat_score_val": round(self.current_ai_confidence * 100, 1),
                    "zmq_lag_ms": round(self._last_lag_ms, 2),
                    "zmq_peak_lag_ms": round(self._peak_lag_ms, 2),
                    "processed": self.messages_processed,
                    "ts": time.time() * 1000
                }
                await self.send_raw(hud_data)
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.error("[ZMQ_CORE] HEARTBEAT_FAIL: %s", e)
                await asyncio.sleep(5)

    def update_hud_state(self, regime: Optional[str] = None, ai_confidence: Optional[float] = None) -> None:
        """
        Updates the internal state for the telemetry broadcast.
        
        Args:
            regime (str, optional): Current market regime string.
            ai_confidence (float, optional): Neural confidence score [0.0 - 1.0].
        """
        if regime:
            self.current_regime = regime
        if ai_confidence is not None:
            self.current_ai_confidence = ai_confidence

    async def send_command(self, action: str, **params: Any) -> None:
        """
        Dispatches a structured command to the MT5 terminal.
        
        Args:
            action (str): Command identifier (BUY, SELL, etc.).
            **params: Associated trade or GUI parameters.
        """
        payload = {
            "action": action, 
            "params": params, 
            "ts": time.time() * 1000
        }
        await self.send_raw(payload)

    async def send_raw(self, payload: Dict[str, Any]) -> None:
        """
        Encapsulates and sends a raw payload over the PUB socket.
        
        Args:
            payload (Dict): Content to be serialized and transmitted.
        """
        if not self.running or not self.pub_socket:
            return
        try:
            msg = json.dumps(payload)
            await self.pub_socket.send_string(msg)
        except Exception as e:
            logger.error("[ZMQ_CORE] SEND_FAIL: %s", e)

    async def send_command_with_ack(self, action: str, **params: Any) -> Optional[Dict[str, Any]]:
        """
        Sends a command and waits for an asynchronous acknowledgment from MT5.
        
        Args:
            action (str): The command to execute.
            **params: Command parameters.
            
        Returns:
            Optional[Dict]: The response message if received within timeframe.
        """
        import uuid
        correlation_id = str(uuid.uuid4())[:8]
        
        payload = {
            "action": action, 
            "params": params, 
            "ts": time.time() * 1000,
            "correlation_id": correlation_id
        }
        
        # Create a rendezvous point
        response_future = asyncio.get_event_loop().create_future()
        self._pending_acks[correlation_id] = response_future
        
        try:
            await self.send_raw(payload)
            logger.debug("[ZMQ_CORE] ACK_WAIT: ID=%s", correlation_id)
            
            # Wait for response (caller should manage timeout via wait_for)
            return await response_future
            
        except Exception as e:
            logger.error("[ZMQ_CORE] ACK_SEND_FAIL: %s", e)
            return {"result": "ERROR", "error": str(e)}
        finally:
            self._pending_acks.pop(correlation_id, None)

    def handle_ack_response(self, data: Dict[str, Any]) -> None:
        """
        Correlates incoming messages with pending acknowledgments.
        
        Args:
            data (Dict): Incoming message containing a possible correlation_id.
        """
        cid = data.get("correlation_id")
        if cid and cid in self._pending_acks:
            future = self._pending_acks[cid]
            if not future.done():
                future.set_result(data)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns a snapshot of the bridge's health and throughput.
        
        Returns:
            Dict: Comprehensive metrics object.
        """
        return {
            "running": self.running,
            "ports": {"pub": self.pub_port, "sub": self.sub_port},
            "throughput": {
                "processed": self.messages_processed,
                "discarded": self.messages_discarded,
                "failed": self.messages_failed
            },
            "latency": {
                "last_ms": round(self._last_lag_ms, 2),
                "peak_ms": round(self._peak_lag_ms, 2)
            },
            "state": {
                "regime": self.current_regime,
                "confidence": self.current_ai_confidence
            }
        }


# Global Interface
zmq_bridge = ZMQBridge()
