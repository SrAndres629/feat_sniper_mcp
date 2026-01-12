import time
import logging
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# Logger configuration
logger = logging.getLogger("MT5_Bridge.Observability")

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
