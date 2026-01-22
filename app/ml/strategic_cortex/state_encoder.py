"""
FEAT SNIPER: STATE ENCODER
==========================
Encodes trading environment state into a fixed-size tensor for the Policy Network.

Gathers information from:
- Account state (balance, phase, risk capital)
- Market microstructure (OFI, entropy, Hurst)
- Neural intelligence (FEAT scores, probabilities)
- Physics validation (Titanium floors, acceleration)
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class StateVector:
    """
    Fixed-dimension state representation for Policy Network.
    
    All values are normalized to approximately [-1, 1] or [0, 1] range
    for stable neural network training.
    """
    # Account State (4 dims)
    balance_normalized: float       # balance / 1000 (clamped [0, 10])
    phase_survival: float           # 1.0 if in survival phase
    phase_consolidation: float      # 1.0 if in consolidation phase
    phase_institutional: float      # 1.0 if in institutional phase
    
    # Market Microstructure (4 dims)
    ofi_z_score: float              # Order flow imbalance z-score [-3, 3]
    entropy_score: float            # Shannon entropy [0, 1]
    hurst_exponent: float           # Hurst H value [0, 1]
    spread_normalized: float        # Current spread / avg spread [0, 2]
    
    # Neural Intelligence (4 dims)
    feat_composite: float           # FEAT chain composite [0, 100] -> [0, 1]
    scalp_prob: float               # Scalp probability [0, 1]
    day_prob: float                 # Day trade probability [0, 1]
    swing_prob: float               # Swing probability [0, 1]
    
    # Physics Validation (4 dims)
    titanium_support: float         # 1.0 if titanium support detected
    titanium_resistance: float      # 1.0 if titanium resistance detected
    acceleration: float             # Kinetic acceleration [-1, 1]
    hurst_gate_valid: float         # 1.0 if Hurst allows EMA signals
    
    # Macro Sentiment (4 dims)
    contrarian_score: float         # [-1, 1] (Short% - Long%) / 100
    retail_long_pct: float          # [0, 1]
    liquidity_above: float          # 1.0 if shorts majority
    liquidity_below: float          # 1.0 if longs majority
    
    # Fractal Mastery (1 dim)
    fractal_coherence: float        # [0, 1] Multi-timeframe alignment summary
    
    # Inter-Temporal Physics (24 dims) - THE MAGIC LINK
    # 8 timeframes * 3 metrics (Direction, Energy, Acceleration)
    temporal_physics: np.ndarray    # Flat array of 24 physics descriptors
    
    def to_tensor(self) -> np.ndarray:
        """Converts state to numpy array for neural network input."""
        return np.array([
            self.balance_normalized,
            self.phase_survival,
            self.phase_consolidation,
            self.phase_institutional,
            self.ofi_z_score,
            self.entropy_score,
            self.hurst_exponent,
            self.spread_normalized,
            self.feat_composite,
            self.scalp_prob,
            self.day_prob,
            self.swing_prob,
            self.titanium_support,
            self.titanium_resistance,
            self.acceleration,
            self.hurst_gate_valid,
            # Macro Sentiment - NEW
            self.contrarian_score,
            self.retail_long_pct,
            self.liquidity_above,
            self.liquidity_below,
            self.fractal_coherence,
            *self.temporal_physics.tolist(),
        ], dtype=np.float32)
    
    @staticmethod
    def get_state_dim() -> int:
        """Returns the dimensionality of the state vector (Institutional V6)."""
        return 24 + 21 # 24 Physics + 21 Meta/Neural


class StateEncoder:
    """
    Encodes raw trading context into normalized StateVector.
    
    This is the 'sensory cortex' that transforms heterogeneous
    signals into a uniform representation for the Policy Network.
    """
    
    def __init__(self):
        self.balance_scale = 1000.0
        self.feat_scale = 100.0
        
    def encode(self,
               account_state: Dict[str, Any],
               microstructure: Dict[str, Any],
               neural_probs: Dict[str, float],
               physics_state: Dict[str, Any],
               sentiment_state: Dict[str, Any] = None,
               fractal_coherence: float = 0.5,
               temporal_physics_dict: Dict[str, float] = None) -> StateVector:
        """
        Encodes all trading context into a StateVector.
        
        Args:
            account_state: {balance, phase_name, risk_capital}
            microstructure: {ofi_z_score, entropy_score, hurst, spread}
            neural_probs: {scalp, day, swing}
            physics_state: {titanium, acceleration, hurst_gate, feat_composite}
            sentiment_state: {contrarian_score, long_pct, liquidity_above, liquidity_below}
            fractal_coherence: Score from FractalCoherenceEngine [0, 1]
            temporal_physics_dict: Map of tf_direction, tf_energy, tf_accel for 8 timeframes
            
        Returns:
            StateVector ready for Policy Network input.
        """
        # Account encoding
        balance = account_state.get("balance", 20.0)
        phase = account_state.get("phase_name", "SURVIVAL")
        
        balance_norm = np.clip(balance / self.balance_scale, 0.0, 10.0)
        phase_survival = 1.0 if "SURVIVAL" in phase.upper() else 0.0
        phase_consolidation = 1.0 if "CONSOLIDATION" in phase.upper() else 0.0
        phase_institutional = 1.0 if "INSTITUTIONAL" in phase.upper() else 0.0
        
        # Microstructure encoding
        ofi_z = np.clip(microstructure.get("ofi_z_score", 0.0), -3.0, 3.0)
        entropy = np.clip(microstructure.get("entropy_score", 0.5), 0.0, 1.0)
        hurst = np.clip(microstructure.get("hurst", 0.5), 0.0, 1.0)
        spread_norm = np.clip(microstructure.get("spread_normalized", 1.0), 0.0, 2.0)
        
        # Neural encoding
        feat_comp = np.clip(physics_state.get("feat_composite", 50.0) / self.feat_scale, 0.0, 1.0)
        p_scalp = np.clip(neural_probs.get("scalp", 0.0), 0.0, 1.0)
        p_day = np.clip(neural_probs.get("day", 0.0), 0.0, 1.0)
        p_swing = np.clip(neural_probs.get("swing", 0.0), 0.0, 1.0)
        
        # Physics encoding (Amplify acceleration as requested for 'preponderante' effect)
        titanium = physics_state.get("titanium", "NEUTRAL")
        ti_support = 1.0 if titanium == "TITANIUM_SUPPORT" else 0.0
        ti_resistance = 1.0 if titanium == "TITANIUM_RESISTANCE" else 0.0
        accel = np.clip(physics_state.get("acceleration", 0.0) * 1.5, -1.0, 1.0) 
        hurst_gate = 1.0 if physics_state.get("hurst_gate", False) else 0.0
        
        # Sentiment encoding (Macro)
        sentiment = sentiment_state or {}
        cont_score = np.clip(sentiment.get("contrarian_score", 0.0), -1.0, 1.0)
        long_pct = np.clip(sentiment.get("long_pct", 50.0) / 100.0, 0.0, 1.0)
        liq_above = float(sentiment.get("liquidity_above", 0.0))
        liq_below = float(sentiment.get("liquidity_below", 0.0))
        
        # Temporal Physics encoding (RIGOROUS NORMALIZATION)
        temp_phys = []
        target_tfs = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
        tpd = temporal_physics_dict or {}
        for tf in target_tfs:
            # Direction is already [-1, 1]
            temp_phys.append(np.clip(tpd.get(f"{tf}_direction", 0.0), -1.0, 1.0))
            # Energy: Needs clamping to avoid hubris
            energy = tpd.get(f"{tf}_energy", 0.0)
            temp_phys.append(np.clip(energy, 0.0, 3.0) / 3.0) # Scale [0, 3] -> [0, 1]
            # Accel: Already tanh normalized in features.py [-1, 1]
            temp_phys.append(np.clip(tpd.get(f"{tf}_accel", 0.0), -1.0, 1.0))
        temp_phys_array = np.array(temp_phys, dtype=np.float32)
        
        return StateVector(
            balance_normalized=float(balance_norm),
            phase_survival=float(phase_survival),
            phase_consolidation=float(phase_consolidation),
            phase_institutional=float(phase_institutional),
            ofi_z_score=float(ofi_z),
            entropy_score=float(entropy),
            hurst_exponent=float(hurst),
            spread_normalized=float(spread_norm),
            feat_composite=float(feat_comp),
            scalp_prob=float(p_scalp),
            day_prob=float(p_day),
            swing_prob=float(p_swing),
            titanium_support=float(ti_support),
            titanium_resistance=float(ti_resistance),
            acceleration=float(accel),
            hurst_gate_valid=float(hurst_gate),
            # Macro Sentiment
            contrarian_score=float(cont_score),
            retail_long_pct=float(long_pct),
            liquidity_above=float(liq_above),
            liquidity_below=float(liq_below),
            # Fractal Mastery
            fractal_coherence=float(fractal_coherence),
            # Inter-Temporal Physics
            temporal_physics=temp_phys_array
        )
    
    def encode_minimal(self, 
                       balance: float,
                       entropy: float,
                       feat_score: float,
                       scalp_prob: float,
                       titanium: bool) -> StateVector:
        """
        Simplified encoder for quick testing.
        """
        phase = "SURVIVAL" if balance < 100 else "CONSOLIDATION" if balance < 500 else "INSTITUTIONAL"
        
        return self.encode(
            account_state={"balance": balance, "phase_name": phase},
            microstructure={"entropy_score": entropy, "ofi_z_score": 0.0, "hurst": 0.5},
            neural_probs={"scalp": scalp_prob, "day": 0.3, "swing": 0.2},
            physics_state={
                "feat_composite": feat_score,
                "titanium": "TITANIUM_SUPPORT" if titanium else "NEUTRAL",
                "acceleration": 0.5,
                "hurst_gate": True
            }
        )


# Singleton
state_encoder = StateEncoder()
