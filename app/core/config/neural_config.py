from pydantic import BaseModel
from typing import Tuple, Dict

class NeuralSettings(BaseModel):
    LAYER_MICRO_PERIODS: Tuple[int, ...] = (3, 5, 8, 13, 21, 34, 55, 89, 144, 233)
    LAYER_OPERATIVE_PERIODS: Tuple[int, ...] = (16, 24, 32, 48, 64, 96, 128, 160, 192, 224)
    LAYER_MACRO_PERIODS: Tuple[int, ...] = (256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280)
    LAYER_BIAS_PERIOD: int = 2048
    KINETIC_COMPRESSION_THRESH: float = 0.3
    KINETIC_EXPANSION_THRESH: float = 0.8
    PHYSICS_MIN_DELTA_T: float = 0.001
    PHYSICS_MAX_VELOCITY: float = 1e6
    PHYSICS_MIN_ATR: float = 1e-8
    PHYSICS_WARMUP_PERIODS: int = 50
    HURST_BUFFER_SIZE: int = 250
    HURST_MIN_SAMPLES: int = 100
    HURST_UPDATE_EVERY_N: int = 50
    LSTM_SEQ_LEN: int = 32
    LSTM_HIDDEN_DIM: int = 64
    LSTM_NUM_LAYERS: int = 2
    NEURAL_INPUT_DIM: int = 24
    NEURAL_HIDDEN_DIM: int = 256
    NEURAL_TCN_CHANNELS: int = 256
    NEURAL_NUM_CLASSES: int = 3
    NEURAL_OUTPUT_HEADS: Tuple[str, ...] = ("logits", "p_win", "volatility", "alpha")
    NEURAL_FEATURE_NAMES: Tuple[str, ...] = (
        "log_ret", "vol_z", "spread_z", "hi_low_ratio", "price_sma_dist",
        "physics_force", "physics_energy", "physics_entropy", "physics_viscosity",
        "structural_feat_index", "confluence_tensor", "proximity_to_structure",
        "vpin_toxicity", "ofi_z", "spread_velocity", "tick_density",
        "align_m5", "align_m15", "align_h1", "align_h4",
        "buy_aggression", "sell_aggression", "delta_vol_z", "absorption_ratio"
    )
    ALPHA_CONFIDENCE_THRESHOLD: float = 0.60
    MC_DROPOUT_SAMPLES: int = 20
    SPATIAL_BINS: int = 50
    STRUCTURAL_PHASE_MAP: Dict[str, float] = {"RANGE": 0.0, "NORMAL": 0.0, "ACCUMULATION": 0.5, "EXPANSION": 1.0, "MOMENTUM": 0.8}
    
    # SMC Configuration
    SMC_SESSION_WEIGHTS: Dict[str, float] = {"LONDON": 1.5, "NY": 1.2, "ASIA": 0.5}
    SMC_OB_MITIGATION_PCT: float = 0.5
    SMC_BOS_THRESHOLD: float = 1.5
    
    # Resonance Engine Configuration (Module 04)
    RESONANCE_HMA_FAST: int = 9       # Hull MA Fast Period
    RESONANCE_HMA_MEDIUM: int = 21    # Hull MA Medium Period
    RESONANCE_HMA_SLOW: int = 55      # Hull MA Slow Period
    RESONANCE_ALMA_PERIOD: int = 200  # ALMA Institutional Anchor
    RESONANCE_ALMA_OFFSET: float = 0.85  # ALMA Gaussian offset (0-1)
    RESONANCE_ALMA_SIGMA: float = 6.0    # ALMA Gaussian sigma
    RESONANCE_ELASTICITY_LOOKBACK: int = 20  # Z-Score lookback
    RESONANCE_DISPERSION_THRESHOLD: float = 2.0  # Std devs for mean reversion
    
    # Temporal Engine Configuration (Module 05)
    # Killzone definitions (UTC hours)
    TEMPORAL_LONDON_OPEN: Tuple[int, int] = (7, 11)    # 07:00 - 11:00 UTC
    TEMPORAL_NY_OPEN: Tuple[int, int] = (13, 17)       # 13:00 - 17:00 UTC (9-1 PM EST)
    TEMPORAL_LONDON_CLOSE: Tuple[int, int] = (15, 17)  # 15:00 - 17:00 UTC (NY/London overlap)
    TEMPORAL_ASIA: Tuple[int, int] = (23, 7)           # 23:00 - 07:00 UTC (low liquidity)
    
    # Session weights (institutional importance)
    TEMPORAL_SESSION_WEIGHTS: Dict[str, float] = {
        "NY_OPEN": 1.0,      # Peak institutional activity
        "LONDON": 0.85,      # High activity
        "LONDON_CLOSE": 0.9, # NY/London overlap = high volatility
        "NY_LATE": 0.5,      # Declining activity
        "ASIA": 0.2,         # Low liquidity, avoid
        "NONE": 0.1          # Dead zone
    }
    
    # Killzone Gaussian parameters
    TEMPORAL_KZ_PEAK_SPREAD: float = 1.5  # Hours around peak for max intensity
    TEMPORAL_FRACTAL_BLOCK_SIZE: int = 4  # 4-hour IPDA blocks
    
    # Training Hyperparameters
    NEURAL_LEARNING_RATE: float = 0.001
    NEURAL_WEIGHT_DECAY: float = 1e-4
    NEURAL_BATCH_SIZE: int = 64
    NEURAL_EPOCHS: int = 50
    NEURAL_LOSS_KINETIC_LAMBDA: float = 0.5
    NEURAL_LOSS_SPATIAL_LAMBDA: float = 0.3
