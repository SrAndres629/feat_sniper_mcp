from __future__ import annotations
from pydantic import BaseModel
from typing import Any

class SystemSettings(BaseModel):
    ZMQ_PORT: int = 5555
    ZMQ_PUB_PORT: int = 5556
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    N8N_API_KEY: str | None = None
    N8N_WEBHOOK_URL: str | None = None
    N8N_URL: str = "http://localhost:5678"
    ENABLE_PROMETHEUS_METRICS: bool = True
    LATENCY_P99_THRESHOLD_MS: int = 200
    DECISION_TTL_MS: int = 1000
    MAX_ORDER_RETRIES: int = 3
    RETRY_BACKOFF_BASE_MS: int = 50
    AUTOML_ENABLED: bool = True
    AUT_EVO_INTERVAL_MINUTES: int = 30
    AUTOML_CHECK_INTERVAL_MINUTES: int = 60
    AUTOML_DRIFT_WIN_RATE_THRESHOLD: float = 0.45
    AUTOML_DRIFT_MIN_TRADES: int = 20
    AUTOML_TRAIN_SCRIPT_PATH: str = "nexus_training/quick_train.py"
    WARMUP_PERIOD_SECONDS: int = 60 # 1 Minute Warmup for ZMQ
    RISK_FACTOR: float = 1.0
    ACCOUNT_CACHE_TTL: float = 0.5
    MARKET_DATA_CACHE_TTL: float = 3.0
