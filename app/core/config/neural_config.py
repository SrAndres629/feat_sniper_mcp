from pydantic import BaseModel
from typing import Tuple, Dict

class NeuralSettings(BaseModel):
    LAYER_MICRO_PERIODS: Tuple[int, ...] = (1, 2, 3, 6, 7, 8, 9, 12, 13, 14)
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
    NEURAL_INPUT_DIM: int = 18
    NEURAL_HIDDEN_DIM: int = 128
    NEURAL_TCN_CHANNELS: int = 64
    NEURAL_NUM_CLASSES: int = 3
    NEURAL_OUTPUT_HEADS: Tuple[str, ...] = ("logits", "p_win", "volatility", "alpha")
    NEURAL_FEATURE_NAMES: Tuple[str, ...] = (
        "dist_micro", "dist_struct", "dist_macro", "dist_bias", 
        "layer_alignment", "kinetic_coherence", "kinetic_pattern_id",
        "dist_poc", "pos_in_va", "density", "energy", "skew", "entropy",
        "form", "space", "accel", "time", "kalman_score"
    )
    ALPHA_CONFIDENCE_THRESHOLD: float = 0.60
    MC_DROPOUT_SAMPLES: int = 20
    SPATIAL_BINS: int = 50
    STRUCTURAL_PHASE_MAP: Dict[str, float] = {"RANGE": 0.0, "NORMAL": 0.0, "ACCUMULATION": 0.5, "EXPANSION": 1.0, "MOMENTUM": 0.8}
    
    # Training Hyperparameters
    NEURAL_LEARNING_RATE: float = 0.001
    NEURAL_WEIGHT_DECAY: float = 1e-4
    NEURAL_BATCH_SIZE: int = 64
    NEURAL_EPOCHS: int = 50
    NEURAL_LOSS_KINETIC_LAMBDA: float = 0.5
    NEURAL_LOSS_SPATIAL_LAMBDA: float = 0.3
