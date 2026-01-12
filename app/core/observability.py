import time
import asyncio
import logging
import functools
from enum import Enum
from typing import Optional, Callable, Any, Dict
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Logger configuration
logger = logging.getLogger("MT5_Bridge.Observability")

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

def resilient(
    max_retries: int = 3, 
    failure_threshold: int = 5, 
    recovery_timeout: int = 30
):
    """
    Decorador de grado industrial para resiliencia (Circuit Breaker + Retries).
    
    Args:
        max_retries: Reintentos inmediatos con jitter.
        failure_threshold: Fallos necesarios para abrir el circuito.
        recovery_timeout: Segundos antes de pasar a HALF_OPEN.
    """
    def decorator(func: Callable):
        state = {
            "status": CircuitState.CLOSED,
            "failures": 0,
            "last_failure_time": 0.0
        }
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if state["status"] == CircuitState.OPEN:
                    if time.time() - state["last_failure_time"] > recovery_timeout:
                        state["status"] = CircuitState.HALF_OPEN
                        logger.info(f"Circuit HALF_OPEN for {func.__name__}")
                    else:
                        raise RuntimeError(f"Circuit is OPEN for {func.__name__}. Fast-failing.")

                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        result = await func(*args, **kwargs)
                        # Reset on success
                        state["status"] = CircuitState.CLOSED
                        state["failures"] = 0
                        return result
                    except Exception as e:
                        last_exception = e
                        state["failures"] += 1
                        state["last_failure_time"] = time.time()
                        
                        if state["failures"] >= failure_threshold:
                            state["status"] = CircuitState.OPEN
                            logger.error(f"Circuit OPENED for {func.__name__} after {state['failures']} failures.")
                        
                        if attempt < max_retries:
                            wait = (2 ** attempt) + (time.time() % 1) # Exponential with jitter
                            logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__} in {wait:.2f}s")
                            await asyncio.sleep(wait)
                
                raise last_exception
            return wrapper
        else:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Synchronous version
                if state["status"] == CircuitState.OPEN:
                    if time.time() - state["last_failure_time"] > recovery_timeout:
                        state["status"] = CircuitState.HALF_OPEN
                    else:
                        raise RuntimeError(f"Circuit is OPEN for {func.__name__}")
                
                last_exception = None
                for attempt in range(max_retries + 1):
                    try:
                        result = func(*args, **kwargs)
                        state["status"] = CircuitState.CLOSED
                        state["failures"] = 0
                        return result
                    except Exception as e:
                        last_exception = e
                        state["failures"] += 1
                        state["last_failure_time"] = time.time()
                        if state["failures"] >= failure_threshold:
                            state["status"] = CircuitState.OPEN
                        if attempt < max_retries:
                            time.sleep(2 ** attempt)
                raise last_exception
            return wrapper
    return decorator

# --- PROMETHEUS METRICS ---
# Latency in seconds
LATENCY_HISTOGRAM = Histogram(
    'mt5_request_latency_seconds', 
    'Latency of MT5 API calls',
    ['operation', 'symbol']
)

# Order tracking
ORDER_COUNTER = Counter(
    'mt5_orders_total', 
    'Total orders sent to MT5',
    ['symbol', 'action', 'status']
)

# Model Health
MODEL_ACCURACY = Gauge(
    'mt5_model_accuracy',
    'Accuracy of the active ML model',
    ['model_name', 'symbol']
)

# --- OPENTELEMETRY SETUP ---
resource = Resource(attributes={
    SERVICE_NAME: "FeatSniper_MCP"
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

class ObservabilityEngine:
    """
    Controlador maestro de telemetra y mtricas institucionales.
    """
    _instance = None
    
    def __init__(self):
        self.start_time = time.time()
        # Iniciar servidor Prometheus en puerto 9090 por defecto
        try:
            start_http_server(9090)
            logger.info("Servidor Prometheus iniciado en puerto 9090.")
        except Exception as e:
            logger.warning(f"No se pudo iniciar Prometheus: {e}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ObservabilityEngine()
        return cls._instance

    def track_latency(self, operation: str, symbol: str, duration: float):
        """Registra la latencia de una operacin."""
        LATENCY_HISTOGRAM.labels(operation=operation, symbol=symbol).observe(duration)

    def track_order(self, symbol: str, action: str, status: str):
        """Registra el resultado de una orden."""
        ORDER_COUNTER.labels(symbol=symbol, action=action, status=status).inc()

    def update_model_health(self, model_name: str, symbol: str, accuracy: float):
        """Actualiza el estado de salud del alpha."""
        MODEL_ACCURACY.labels(model_name=model_name, symbol=symbol).set(accuracy)

obs_engine = ObservabilityEngine.get_instance()
