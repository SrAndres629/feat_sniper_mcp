from pydantic import BaseModel
from typing import Optional

class SystemSettings(BaseModel):
    ZMQ_PORT: int = 5555
    ZMQ_PUB_PORT: int = 5556
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    N8N_API_KEY: Optional[str] = None
    N8N_WEBHOOK_URL: Optional[str] = None
    N8N_URL: str = "http://localhost:5678"
    ENABLE_PROMETHEUS_METRICS: bool = True
    LATENCY_P99_THRESHOLD_MS: int = 200
    DECISION_TTL_MS: int = 1000
    MAX_ORDER_RETRIES: int = 3
    RETRY_BACKOFF_BASE_MS: int = 50
    AUTOML_ENABLED: bool = True
    AUT_EVO_INTERVAL_MINUTES: int = 30
