from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Configuración global del sistema cargada desde variables de entorno o archivo .env.
    """
    # MetaTrader 5 Credentials
    MT5_LOGIN: int
    MT5_PASSWORD: str
    MT5_SERVER: str
    MT5_PATH: Optional[str] = None  # Ruta al terminal64.exe si no es la por defecto
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    N8N_API_KEY: Optional[str] = None
    
    # Security & Risk Management
    MAX_DAILY_DRAWDOWN_PERCENT: float = 3.0
    RISK_FREE_MODE: bool = False  # If True, limits operations for testing
    
    # Cache Settings
    MARKET_DATA_CACHE_TTL: int = 2  # Segundos
    ACCOUNT_CACHE_TTL: float = 0.5  # Segundos
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instancia global de configuración
settings = Settings()
