from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """
    Configuración global del sistema cargada desde variables de entorno o archivo .env.
    """
    # MetaTrader 5 Credentials
    # MetaTrader 5 Credentials (Optional for Linux/Docker)
    MT5_LOGIN: Optional[int] = None
    MT5_PASSWORD: Optional[str] = None
    MT5_SERVER: Optional[str] = None
    MT5_PATH: Optional[str] = None  # Ruta al terminal64.exe si no es la por defecto
    
    # Asset Configuration
    SYMBOL: str = "BTCUSD"
    
    # Connectivity Ports
    ZMQ_PORT: int = 5555
    
    # API Settings
    API_HOST: str = "127.0.0.1"  # Security: Bind to localhost by default
    API_PORT: int = 8000
    DEBUG: bool = False
    N8N_API_KEY: Optional[str] = None
    
    # Version 2.0 Institutional Extensions
    ML_MODEL_PATH: Optional[str] = "models/setup_classifier.pkl"
    MIN_LIQUIDITY_DEPTH: float = 1000000.0  # USD equivalent in DoM
    ENABLE_CIRCUIT_BREAKER: bool = True
    CB_FAILURE_THRESHOLD: int = 5
    MESSAGING_BROKER_URL: Optional[str] = None  # For Kafka/RabbitMQ future scaling
    
    # Advanced Risk Management
    MAX_DAILY_DRAWDOWN_PERCENT: float = 3.0
    RISK_PER_TRADE_PERCENT: float = 1.0
    MAX_OPEN_POSITIONS: int = 20  # Increased for multi-asset strategies
    MAX_CORRELATION_LIMIT: float = 0.65 
    VOLATILITY_ADAPTIVE_LOTS: bool = True
    
    # Institutional Telemetry 2.0
    ENABLE_PROMETHEUS_METRICS: bool = True
    LATENCY_P99_THRESHOLD_MS: int = 200
    CORRELATION_ID_HEADER: str = "X-Correlation-ID"
    
    # Execution Control (Safety)
    EXECUTION_ENABLED: bool = False  # Master Switch
    SHADOW_MODE: bool = True         # Simulation Mode (Paper Trading)
    
    # Supabase & Cloud Nexus
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    
    # SSH Offloading (Remote Cómputo)
    SSH_HOST: Optional[str] = None
    SSH_USER: Optional[str] = None
    SSH_KEY_PATH: Optional[str] = None
    ENABLE_SSH_OFFLOADING: bool = False
    
    # Cache Settings
    MARKET_DATA_CACHE_TTL: int = 1  # Professional grade: Lower TTL
    ACCOUNT_CACHE_TTL: float = 0.5 
    
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8", 
        extra="ignore",
        case_sensitive=False
    )

# Instancia global de configuración
settings = Settings()
