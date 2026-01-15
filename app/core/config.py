"""
FEAT NEXUS - Master Configuration Contract
============================================
Pydantic-enforced configuration with institutional-grade validation.

[P0 REFACTOR] Changes:
- Immutable tuples for FEAT Layers
- @field_validator for critical parameters
- @model_validator for cross-field consistency
- Windows enforcement for MT5 credentials
- ML constants centralized (Single Source of Truth)
"""
import sys
import json
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import Optional, Tuple
from pathlib import Path


class Settings(BaseSettings):
    """
    Configuración global del sistema cargada desde variables de entorno o archivo .env.
    
    Esta clase es el CONTRATO MAESTRO del sistema. Cualquier parámetro crítico
    debe estar definido aquí para cumplir con Single Source of Truth.
    """
    
    # =========================================================================
    # METTRADER 5 CREDENTIALS
    # =========================================================================
    MT5_LOGIN: Optional[int] = None
    MT5_PASSWORD: Optional[str] = None
    MT5_SERVER: Optional[str] = None
    MT5_PATH: Optional[str] = None  # Ruta al terminal64.exe si no es la default
    
    # =========================================================================
    # ASSET CONFIGURATION
    # =========================================================================
    SYMBOL: str = "BTCUSD"
    
    # =========================================================================
    # CONNECTIVITY PORTS
    # =========================================================================
    ZMQ_PORT: int = 5555
    ZMQ_PUB_PORT: int = 5556
    API_HOST: str = "127.0.0.1"  # Security: Bind to localhost by default
    API_PORT: int = 8000
    
    # =========================================================================
    # FEAT LAYER CONFIGURATION (IMMUTABLE TUPLES)
    # =========================================================================
    # [P0 FIX] Changed from list to tuple for immutability
    LAYER_MICRO_PERIODS: Tuple[int, ...] = (1, 2, 3, 6, 7, 8, 9, 12, 13, 14)
    LAYER_OPERATIVE_PERIODS: Tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128, 160, 192, 224)
    LAYER_MACRO_PERIODS: Tuple[int, ...] = (256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280)
    LAYER_BIAS_PERIOD: int = 2048  # Layer 4: Gravity (SMMA)
    
    # =========================================================================
    # ML/NEURAL NETWORK CONFIGURATION (Single Source of Truth)
    # =========================================================================
    # [P1 FIX] Centralized from ml_engine.py and alpha_engine.py
    LSTM_SEQ_LEN: int = 32              # Sequence length for LSTM input
    LSTM_HIDDEN_DIM: int = 64           # Hidden dimension of LSTM
    LSTM_NUM_LAYERS: int = 2            # Number of LSTM layers
    HURST_BUFFER_SIZE: int = 250        # Buffer size for Hurst calculation
    HURST_MIN_SAMPLES: int = 100        # Minimum samples before computing Hurst
    HURST_UPDATE_EVERY_N: int = 50      # Recalculate Hurst every N ticks
    SHARPE_WINDOW_SIZE: int = 50        # Rolling window for Sharpe calculation
    SHARPE_MIN_TRADES: int = 10         # Minimum trades before adaptive weights
    ALPHA_CONFIDENCE_THRESHOLD: float = 0.60  # Minimum confidence to execute trade
    ENSEMBLE_MIN_WEIGHT: float = 0.15   # Minimum weight for any model
    ENSEMBLE_MAX_WEIGHT: float = 0.85   # Maximum weight for any model
    
    # =========================================================================
    # API SETTINGS
    # =========================================================================
    DEBUG: bool = False
    N8N_API_KEY: Optional[str] = None
    N8N_WEBHOOK_URL: Optional[str] = None
    
    # =========================================================================
    # VERSION 2.0 INSTITUTIONAL EXTENSIONS
    # =========================================================================
    ML_MODEL_PATH: Optional[str] = "models/setup_classifier.pkl"
    MIN_LIQUIDITY_DEPTH: float = 1000000.0  # USD equivalent in DoM
    ENABLE_CIRCUIT_BREAKER: bool = True
    CB_FAILURE_THRESHOLD: int = 5
    MESSAGING_BROKER_URL: Optional[str] = None  # For Kafka/RabbitMQ future scaling
    
    # =========================================================================
    # ADVANCED RISK MANAGEMENT
    # =========================================================================
    MAX_DAILY_DRAWDOWN_PERCENT: float = 6.0  # Max hard limit (Order 66)
    CB_LEVEL_1_DD: float = 2.0  # 2% DD -> 75% Lot Size (25% Reduction)
    CB_LEVEL_2_DD: float = 4.0  # 4% DD -> 50% Lot Size + 1h Pause
    CB_LEVEL_3_DD: float = 6.0  # 6% DD -> Total Shutdown (HALT)
    RISK_PER_TRADE_PERCENT: float = 1.0
    MAX_OPEN_POSITIONS: int = 20  # Increased for multi-asset strategies
    MAX_CORRELATION_LIMIT: float = 0.65 
    VOLATILITY_ADAPTIVE_LOTS: bool = True
    ATR_TRAILING_MULTIPLIER: float = 1.5
    PHANTOM_MODE_ENABLED: bool = True
    
    # =========================================================================
    # TWIN-ENGINE HYBRID STRATEGY
    # =========================================================================
    MAGIC_SCALP: int = 234001
    MAGIC_SWING: int = 234002
    EQUITY_UNLOCK_THRESHOLD: float = 50.0  # USD to unlock 3rd position
    SCALP_TARGET_USD: float = 2.0
    SWING_TARGET_USD: float = 10.0
    INITIAL_CAPITAL: float = 20.0
    
    # =========================================================================
    # INSTITUTIONAL TELEMETRY 2.0
    # =========================================================================
    ENABLE_PROMETHEUS_METRICS: bool = True
    LATENCY_P99_THRESHOLD_MS: int = 200
    CORRELATION_ID_HEADER: str = "X-Correlation-ID"
    
    # =========================================================================
    # EXECUTION CONTROL (SAFETY)
    # =========================================================================
    EXECUTION_ENABLED: bool = False  # Master Switch
    SHADOW_MODE: bool = True         # Simulation Mode (Paper Trading)
    DRY_RUN: bool = False            # LIVE SYSTEM ACTIVE (Model 5 Launch)
    
    # =========================================================================
    # LOW-LATENCY EXECUTION SETTINGS
    # =========================================================================
    DECISION_TTL_MS: int = 300        # Max age of ML decision before rejection (ms)
    MAX_ORDER_RETRIES: int = 3        # Retry attempts for transient errors (REQUOTE)
    RETRY_BACKOFF_BASE_MS: int = 50   # Exponential backoff base (50 -> 150 -> 450ms)
    
    # =========================================================================
    # PHYSICAL GUARDIANS (The Quant Flow - Alpha Optimization)
    # =========================================================================
    VOLATILITY_THRESHOLD: float = 3.0    # ATR % Threshold (Flash Crash Guard)
    SPREAD_MAX_PIPS: float = 50          # Max Spread in Points/Pips (Liquidity Guard)
    SPREAD_MULTIPLIER_MAX: float = 3.0   # Max spread as multiple of average
    
    # =========================================================================
    # SUPABASE & CLOUD NEXUS
    # =========================================================================
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    
    # =========================================================================
    # SSH OFFLOADING (Remote Compute)
    # =========================================================================
    SSH_HOST: Optional[str] = None
    SSH_USER: Optional[str] = None
    SSH_KEY_PATH: Optional[str] = None
    ENABLE_SSH_OFFLOADING: bool = False
    
    # =========================================================================
    # CACHE SETTINGS
    # =========================================================================
    MARKET_DATA_CACHE_TTL: int = 1    # Professional grade: Lower TTL
    ACCOUNT_CACHE_TTL: float = 0.5 
    
    # =========================================================================
    # VALIDATORS
    # =========================================================================
    
    @field_validator('MT5_LOGIN', mode='before')
    @classmethod
    def validate_mt5_login_on_windows(cls, v):
        """
        [P0] Enforce MT5_LOGIN on Windows platform.
        In Docker/Linux, MT5 is not available so credentials are optional.
        """
        if sys.platform == 'win32' and v is None:
            # Allow boot but log loud warning - don't block startup
            import logging
            logging.warning(
                "[CONFIG] ⚠️ MT5_LOGIN no definido en Windows. "
                "El sistema operará en modo degradado sin conexión MT5."
            )
        return v
    
    @field_validator('RISK_PER_TRADE_PERCENT', mode='after')
    @classmethod
    def validate_risk_cap(cls, v):
        """
        [P0] Institutional risk cap: Never exceed 5% per trade.
        """
        if v > 5.0:
            raise ValueError(
                f"RISK_PER_TRADE_PERCENT={v}% excede el límite institucional de 5%. "
                "Ajusta en el archivo .env para evitar blow-up."
            )
        if v <= 0:
            raise ValueError("RISK_PER_TRADE_PERCENT debe ser positivo.")
        return v
    
    @field_validator('SPREAD_MAX_PIPS', mode='after')
    @classmethod
    def validate_spread_positive(cls, v):
        """
        [P0] Spread cannot be negative.
        """
        if v < 0:
            raise ValueError("SPREAD_MAX_PIPS no puede ser negativo.")
        return v
    
    @field_validator('DECISION_TTL_MS', mode='after')
    @classmethod
    def validate_ttl_positive(cls, v):
        """
        [P0] TTL=0 would discard all messages.
        """
        if v <= 0:
            raise ValueError("DECISION_TTL_MS debe ser mayor a 0.")
        return v
    
    @field_validator('LAYER_MICRO_PERIODS', 'LAYER_OPERATIVE_PERIODS', 'LAYER_MACRO_PERIODS', mode='before')
    @classmethod
    def parse_layer_periods(cls, v):
        """
        [P1] Parse JSON string from env or convert list to tuple.
        Enables Bayesian Auto-Tuning via environment variables.
        """
        if isinstance(v, str):
            return tuple(json.loads(v))
        if isinstance(v, list):
            return tuple(v)
        return v
    
    @model_validator(mode='after')
    def validate_circuit_breaker_hierarchy(self):
        """
        [P0] Ensure CB levels are properly ordered: L1 < L2 < L3 <= MAX_DD
        """
        if not (self.CB_LEVEL_1_DD < self.CB_LEVEL_2_DD < self.CB_LEVEL_3_DD <= self.MAX_DAILY_DRAWDOWN_PERCENT):
            raise ValueError(
                f"Circuit Breaker hierarchy invalid: "
                f"L1({self.CB_LEVEL_1_DD}) < L2({self.CB_LEVEL_2_DD}) < "
                f"L3({self.CB_LEVEL_3_DD}) <= MAX({self.MAX_DAILY_DRAWDOWN_PERCENT})"
            )
        return self
    
    @model_validator(mode='after')
    def validate_port_collision(self):
        """
        [P0] ZMQ_PORT and API_PORT cannot be the same.
        """
        if self.ZMQ_PORT == self.API_PORT:
            raise ValueError(
                f"Port collision detected: ZMQ_PORT={self.ZMQ_PORT} == API_PORT={self.API_PORT}. "
                "Both cannot use the same port."
            )
        return self
    
    @model_validator(mode='after')
    def validate_hurst_samples(self):
        """
        [P1] Hurst MIN_SAMPLES must be <= BUFFER_SIZE
        """
        if self.HURST_MIN_SAMPLES > self.HURST_BUFFER_SIZE:
            raise ValueError(
                f"HURST_MIN_SAMPLES ({self.HURST_MIN_SAMPLES}) cannot exceed "
                f"HURST_BUFFER_SIZE ({self.HURST_BUFFER_SIZE})."
            )
        return self
    
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8", 
        extra="ignore",
        case_sensitive=False
    )


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================
settings = Settings()
