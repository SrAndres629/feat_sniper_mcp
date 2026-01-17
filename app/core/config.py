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
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator, computed_field
from typing import List, Dict, Any, Optional, Tuple, Literal
from pathlib import Path


class ExecutionMode(str, Enum):
    """Execution context for the trading system."""

    LIVE = "LIVE"  # Real money execution
    PAPER = "PAPER"  # Paper trading (demo account)
    SHADOW = "SHADOW"  # Signal generation only, no execution
    BACKTEST = "BACKTEST"  # Historical simulation


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
    LAYER_OPERATIVE_PERIODS: Tuple[int, ...] = (
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        160,
        192,
        224,
    )
    LAYER_MACRO_PERIODS: Tuple[int, ...] = (
        256,
        320,
        384,
        448,
        512,
        640,
        768,
        896,
        1024,
        1280,
    )
    LAYER_BIAS_PERIOD: int = 2048  # Layer 4: Gravity (SMMA)
    
    # [LEVEL 49] KINETIC PATTERN THRESHOLDS
    KINETIC_COMPRESSION_THRESH: float = 0.3 # Knot Threshold
    KINETIC_EXPANSION_THRESH: float = 0.8  # Fan Threshold

    # =========================================================================
    # ML/NEURAL NETWORK CONFIGURATION (Single Source of Truth)
    # =========================================================================
    # [P1 FIX] Centralized from ml_engine.py and alpha_engine.py
    LSTM_SEQ_LEN: int = 32  # Sequence length for LSTM input
    LSTM_HIDDEN_DIM: int = 64  # Hidden dimension of LSTM
    LSTM_NUM_LAYERS: int = 2  # Number of LSTM layers
    HURST_BUFFER_SIZE: int = 250  # Buffer size for Hurst calculation
    HURST_MIN_SAMPLES: int = 100  # Minimum samples before computing Hurst
    HURST_UPDATE_EVERY_N: int = 50  # Recalculate Hurst every N ticks
    SHARPE_WINDOW_SIZE: int = 50  # Rolling window for Sharpe calculation
    SHARPE_MIN_TRADES: int = 10  # Minimum trades before adaptive weights
    ALPHA_CONFIDENCE_THRESHOLD: float = 0.60  # Minimum confidence to execute trade
    ENSEMBLE_MIN_WEIGHT: float = 0.15  # Minimum weight for any model
    ENSEMBLE_MAX_WEIGHT: float = 0.85  # Maximum weight for any model
    
    # [P0 RESTORED] Institutional Safety Constants
    SPREAD_MAX_PIPS: float = 50.0
    ATR_SL_MULTIPLIER: float = 2.0
    ATR_TP_MULTIPLIER: float = 4.0
    DECISION_TTL_MS: int = 1000

    # [LEVEL 56] DOCTORAL NEURAL ARCHITECTURE
    NEURAL_INPUT_DIM: int = 18  # Total features (Temporal + Structural + Kinetic)
    NEURAL_HIDDEN_DIM: int = 128
    NEURAL_TCN_CHANNELS: int = 64
    NEURAL_NUM_CLASSES: int = 3  # [SELL, HOLD, BUY]
    NEURAL_OUTPUT_HEADS: Tuple[str, ...] = ("logits", "p_win", "volatility", "alpha")
    MC_DROPOUT_SAMPLES: int = 20  # For Epistemic Uncertainty Estimation
    SPATIAL_BINS: int = 50  # 50x50 Vision Resolution
    
    # [LEVEL 61] NEURAL FEATURE SET (18 DIMENSIONS)
    NEURAL_FEATURE_NAMES: Tuple[str, ...] = (
        "dist_micro", "dist_struct", "dist_macro", "dist_bias", 
        "layer_alignment", "kinetic_coherence", "kinetic_pattern_id",
        "dist_poc", "pos_in_va", "density", "energy", "skew", "entropy",
        "form", "space", "accel", "time", "kalman_score"
    )

    # =========================================================================
    # API SETTINGS
    # =========================================================================
    DEBUG: bool = False
    N8N_API_KEY: Optional[str] = None
    N8N_WEBHOOK_URL: Optional[str] = None
    N8N_URL: str = "http://localhost:5678"  # Base URL for n8n auto-discovery

    # =========================================================================
    # HEADLESS MODE (Linux/Docker without MT5)
    # =========================================================================
    HEADLESS_MODE: bool = False  # If True, skip MT5 connection (for CI/CD, testing)

    # =========================================================================
    # VERSION 2.0 INSTITUTIONAL EXTENSIONS
    # =========================================================================
    MODELS_DIR: str = "models" # [P1 FIX] Directory for saved models
    ML_MODEL_PATH: Optional[str] = "models/setup_classifier.pkl"
    MIN_LIQUIDITY_DEPTH: float = 1000000.0  # USD equivalent in DoM
    ENABLE_CIRCUIT_BREAKER: bool = True
    CB_FAILURE_THRESHOLD: int = 5
    MESSAGING_BROKER_URL: Optional[str] = None  # For Kafka/RabbitMQ future scaling
    
    # [LEVEL 61] AUTOML SOVEREIGN CONFIG
    AUTOML_ENABLED: bool = True
    AUT_EVO_INTERVAL_MINUTES: int = 30
    AUTOML_DRIFT_WIN_RATE_THRESHOLD: float = 0.55
    AUTOML_DRIFT_BIAS_THRESHOLD: float = 0.15
    AUTOML_DRIFT_MIN_TRADES: int = 30
    AUTOML_TRAIN_SCRIPT_PATH: str = "nexus_training/train_hybrid.py"
    
    # [LEVEL 61] UNSUPERVISED REGIME & ANOMALY CONSTANTS
    REGIME_CLUSTERS: int = 3
    REGIME_MODEL_PATH: str = "models/kmeans_regime.pkl"
    REGIME_SCALER_PATH: str = "models/kmeans_scaler.pkl"
    ANOMALY_MODEL_PATH: str = "models/isolation_forest.pkl"
    ANOMALY_CONTAMINATION: float = 0.05
    AUTOML_MIN_TRADES_FOR_DRIFT: int = 30
    
    # [LEVEL 61] INSTITUTIONAL STRUCTURAL SETTINGS
    STRUCTURAL_PHASE_MAP: Dict[str, float] = {
        "RANGE": 0.0, "NORMAL": 0.0, "ACCUMULATION": 0.5, "EXPANSION": 1.0, "MOMENTUM": 0.8
    }
    ZONE_PROXIMITY_FACTOR: float = 0.0005 # Factor of price
    
    # [LEVEL 61] MARKET PHYSICS THRESHOLDS
    PHYSICS_INITIATIVE_VOL_THRESHOLD: float = 2.5
    PHYSICS_INITIATIVE_VEL_THRESHOLD: float = 1.5
    PHYSICS_REGIME_SIGMA: float = 2.0
    
    # [LEVEL 61] CONVERGENCE ENGINE THRESHOLDS
    CONVERGENCE_MIN_SCORE: float = 0.75
    CONVERGENCE_MAX_UNCERTAINTY: float = 0.04
    CONVERGENCE_MIN_KINETIC_COHERENCE: float = 0.6

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

    # =========================================================================
    # PERFORMANCE & LOGGING (Operación Navaja)
    # =========================================================================
    PERFORMANCE_MODE: bool = True  # Disables non-essential live cloud sync
    TRADING_MODE: Literal["LIVE", "PAPER", "SHADOW", "BACKTEST"] = "SHADOW"
    VOLATILITY_ADAPTIVE_LOTS: bool = True
    ATR_TRAILING_MULTIPLIER: float = 1.5
    PHANTOM_MODE_ENABLED: bool = True

    # =========================================================================
    # CONTEXT-AWARE COMPUTED PROPERTIES (The Reactor)
    # =========================================================================
    @computed_field
    @property
    def execution_mode(self) -> ExecutionMode:
        """Unified execution mode from various legacy flags."""
        if (
            self.TRADING_MODE == "LIVE"
            and self.EXECUTION_ENABLED
            and not self.SHADOW_MODE
        ):
            return ExecutionMode.LIVE
        elif self.TRADING_MODE == "PAPER" or (
            self.SHADOW_MODE and self.EXECUTION_ENABLED
        ):
            return ExecutionMode.PAPER
        elif self.TRADING_MODE == "BACKTEST":
            return ExecutionMode.BACKTEST
        return ExecutionMode.SHADOW

    # [LEVEL 57] DOCTORAL PRODUCTION LOCKDOWN
    MT5_MAGIC_NUMBER: int = 123456
    MT5_ORDER_COMMENT: str = "FEAT_AI_Sniper_V2"
    
    # Trade Management Invariants
    WARMUP_PERIOD_SECONDS: int = 60
    EXHAUSTION_ATR_THRESHOLD: float = 0.5
    SCALP_TIME_LIMIT_SECONDS: int = 300
    
    # Physics Invariants
    PHYSICS_MIN_DELTA_T: float = 0.001
    PHYSICS_MAX_VELOCITY: float = 1e6
    PHYSICS_MIN_ATR: float = 1e-8
    PHYSICS_WARMUP_PERIODS: int = 50
    
    # Session Killzones (UTC-4 Reference)
    KILLZONE_NY_START: int = 7
    KILLZONE_NY_END: int = 11
    KILLZONE_LONDON_START: int = 3
    KILLZONE_LONDON_END: int = 5
    KILLZONE_ASIA_START: int = 20
    KILLZONE_ASIA_END: int = 0  # Midnight wrap
    UTC_OFFSET: int = -4

    @computed_field
    @property
    def is_live_trading(self) -> bool:
        """True if system will execute real money trades."""
        return self.execution_mode == ExecutionMode.LIVE

    @computed_field
    @property
    def effective_risk_cap(self) -> float:
        """Dynamic risk cap based on execution context."""
        if self.is_live_trading:
            return min(self.RISK_PER_TRADE_PERCENT, 2.0)  # Hard 2% cap for live
        return self.RISK_PER_TRADE_PERCENT  # Use configured value for paper/shadow

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
    SHADOW_MODE: bool = True  # Simulation Mode (Paper Trading)
    DRY_RUN: bool = False  # LIVE SYSTEM ACTIVE (Model 5 Launch)
    HEADLESS_MODE: bool = False  # True = allow no MT5 (Linux/Docker)

    # =========================================================================
    # LOW-LATENCY EXECUTION SETTINGS
    # =========================================================================
    DECISION_TTL_MS: int = 1000  # Increased for ML processing slack (Audit Fix)
    MAX_ORDER_RETRIES: int = 3  # Retry attempts for transient errors (REQUOTE)
    RETRY_BACKOFF_BASE_MS: int = 50  # Exponential backoff base (50 -> 150 -> 450ms)

    # =========================================================================
    # PHYSICAL GUARDIANS (The Quant Flow - Alpha Optimization)
    # =========================================================================
    VOLATILITY_THRESHOLD: float = 3.0  # ATR % Threshold (Flash Crash Guard)
    SPREAD_MAX_PIPS: float = 50  # Max Spread in Points/Pips (Liquidity Guard)
    SPREAD_MULTIPLIER_MAX: float = 3.0  # Max spread as multiple of average

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
    MARKET_DATA_CACHE_TTL: int = 1  # Professional grade: Lower TTL
    ACCOUNT_CACHE_TTL: float = 0.5

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator("MT5_LOGIN", mode="before")
    @classmethod
    def validate_mt5_login_on_windows(cls, v):
        """
        [P0] Enforce MT5_LOGIN on Windows platform.
        In Docker/Linux, MT5 is not available so credentials are optional.
        """
        if sys.platform == "win32" and v is None:
            # Allow boot but log loud warning - don't block startup
            import logging

            logging.warning(
                "[CONFIG] ⚠️ MT5_LOGIN no definido en Windows. "
                "El sistema operará en modo degradado sin conexión MT5."
            )
        return v

    @field_validator("RISK_PER_TRADE_PERCENT", mode="after")
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

    @field_validator("SPREAD_MAX_PIPS", mode="after")
    @classmethod
    def validate_spread_positive(cls, v):
        """
        [P0] Spread cannot be negative.
        """
        if v < 0:
            raise ValueError("SPREAD_MAX_PIPS no puede ser negativo.")
        return v

    @field_validator("ATR_TRAILING_MULTIPLIER", "ATR_SL_MULTIPLIER", "ATR_TP_MULTIPLIER", mode="after")
    @classmethod
    def validate_atr_multipliers(cls, v):
        """
        [LEVEL 58] Validation of institutional ATR multiples to prevent anomalous behavior.
        """
        if v < 0.5 or v > 15.0:
            raise ValueError(f"Multiplicador ATR ({v}) fuera de rango institucional [0.5, 15.0].")
        return v

    @field_validator("DECISION_TTL_MS", mode="after")
    @classmethod
    def validate_ttl_positive(cls, v):
        """
        [P0] TTL=0 would discard all messages.
        """
        if v <= 0:
            raise ValueError("DECISION_TTL_MS debe ser mayor a 0.")
        return v

    @field_validator(
        "LAYER_MICRO_PERIODS",
        "LAYER_OPERATIVE_PERIODS",
        "LAYER_MACRO_PERIODS",
        mode="before",
    )
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

    @model_validator(mode="after")
    def validate_circuit_breaker_hierarchy(self):
        """
        [P0] Ensure CB levels are properly ordered: L1 < L2 < L3 <= MAX_DD
        """
        if not (
            self.CB_LEVEL_1_DD
            < self.CB_LEVEL_2_DD
            < self.CB_LEVEL_3_DD
            <= self.MAX_DAILY_DRAWDOWN_PERCENT
        ):
            raise ValueError(
                f"Circuit Breaker hierarchy invalid: "
                f"L1({self.CB_LEVEL_1_DD}) < L2({self.CB_LEVEL_2_DD}) < "
                f"L3({self.CB_LEVEL_3_DD}) <= MAX({self.MAX_DAILY_DRAWDOWN_PERCENT})"
            )
        return self

    @model_validator(mode="after")
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

    @model_validator(mode="after")
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

    @model_validator(mode="after")
    def load_evolutionary_overrides(self):
        """
        [LEVEL 30] AUTOML INJECTION.
        Loads 'config/dynamic_params.json' if it exists and overrides defaults.
        This allows the EvolutionaryOptimizer to tune the system live.
        """
        try:
            dynamic_path = Path(__file__).parent.parent.parent / "config" / "dynamic_params.json"
            if dynamic_path.exists():
                text = dynamic_path.read_text(encoding="utf-8")
                if not text.strip():
                    return self
                    
                overrides = json.loads(text)
                
                # Apply overrides if they match existing fields
                for key, value in overrides.items():
                    if hasattr(self, key):
                        # Attempt to cast to the type of the target attribute
                        target_type = type(getattr(self, key))
                        try:
                            setattr(self, key, target_type(value))
                        except (ValueError, TypeError):
                            setattr(self, key, value)
        except Exception as e:
            # Fallback to defaults
            pass
        return self

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================
settings = Settings()
