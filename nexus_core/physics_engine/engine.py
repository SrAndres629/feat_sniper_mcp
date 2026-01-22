import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from app.core.config import settings

logger = logging.getLogger("Nexus.PhysicsEngine")

class KineticValidator:
    """
    [DOCTORAL SKILL: ABSORPTION TEST]
    Validates if an Institutional Impulse is real or a fakeout.
    """
    def __init__(self, epsilon: float = 1e-9):
        self.eps = epsilon

    def calculate_feat_force(self, candle: pd.Series, atr: float, rvol: float) -> float:
        """Force = (Body_Size / (ATR + EPS)) * Relative_Volume"""
        body_size = abs(candle["close"] - candle["open"])
        return (body_size / (atr + self.eps)) * rvol

    def check_absorption_state(self, df: pd.DataFrame, limit_pct: float = 0.5, window: int = 3) -> Dict[str, Any]:
        """State Machine for Absorption Logic."""
        if len(df) < window + 2: 
            return {"state": "NEUTRAL", "progress": 0, "feat_force": 0.0}
        
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        if atr <= 0: atr = self.eps
        
        active_impulse_idx = None
        for i in range(0, window + 1): 
            idx = -(i + 1)
            candle = df.iloc[idx]
            vol_mean = df["volume"].iloc[idx-20:idx].mean() if abs(idx) > 20 else df["volume"].mean()
            rvol = candle["volume"] / (vol_mean + self.eps)
            force = self.calculate_feat_force(candle, atr, rvol)
            if force > 2.0:
                active_impulse_idx = idx
                break
        
        if active_impulse_idx is None:
             curr = df.iloc[-1]
             vol_mean = df["volume"].iloc[-21:-1].mean()
             rvol = curr["volume"] / (vol_mean + self.eps)
             force = self.calculate_feat_force(curr, atr, rvol)
             return {"state": "IMPULSE" if force > 2.0 else "NEUTRAL", "progress": 0, "feat_force": force}

        impulse_candle = df.iloc[active_impulse_idx]
        impulse_body = abs(impulse_candle["close"] - impulse_candle["open"])
        is_bull = impulse_candle["close"] > impulse_candle["open"]
        limit_level = impulse_candle["low"] + (impulse_body * (1.0 - limit_pct)) if is_bull else impulse_candle["high"] - (impulse_body * (1.0 - limit_pct))
        
        candles_passed = abs(active_impulse_idx) - 1
        subsequent = df.iloc[active_impulse_idx+1:]
        
        for _, fut in subsequent.iterrows():
            if (is_bull and fut["close"] < limit_level) or (not is_bull and fut["close"] > limit_level):
                return {"state": "FAILED", "progress": candles_passed, "feat_force": 0.0}
                    
        if candles_passed >= window:
            return {"state": "CONFIRMED", "progress": window, "feat_force": 0.0}
            
        return {"state": "MONITORING", "progress": candles_passed, "feat_force": 0.0}

class SpectralMechanics:
    """[LEVEL 51] SPECTRAL TENSOR ENCODING"""
    def __init__(self, epsilon: float = 1e-9):
        self.eps = epsilon
        self.sub_1 = settings.LAYER_MICRO_PERIODS[:3]
        self.sub_2 = settings.LAYER_MICRO_PERIODS[3:7]
        self.sub_3 = settings.LAYER_MICRO_PERIODS[7:]
        self.sub_4 = settings.LAYER_OPERATIVE_PERIODS[:3]
        self.sub_5 = settings.LAYER_OPERATIVE_PERIODS[3:6]
        self.sub_6 = settings.LAYER_OPERATIVE_PERIODS[6:]
        self.sub_7 = settings.LAYER_MACRO_PERIODS[:3]
        self.sub_8 = settings.LAYER_MACRO_PERIODS[3:6]
        self.sub_9 = settings.LAYER_MACRO_PERIODS[6:]
        self.sub_10 = [settings.LAYER_BIAS_PERIOD]

    def compute_group_integrity(self, emas: List[float]) -> float:
        if len(emas) < 2: return 0.0
        bull_pairs = sum(1 for i in range(len(emas)-1) if emas[i] > emas[i+1])
        bear_pairs = sum(1 for i in range(len(emas)-1) if emas[i] < emas[i+1])
        total_pairs = len(emas) - 1
        return (bull_pairs - bear_pairs) / total_pairs if total_pairs > 0 else 0.0

    def compute_chromatic_divergence(self, emas: List[float], atr: float) -> float:
        if not emas: return 0.0
        return (max(emas) - min(emas)) / (atr + self.eps)

    def analyze_spectrum(self, close_series: pd.Series, atr: float = 1.0) -> Dict[str, float]:
        metrics = {}
        def get_emas(periods):
            return [close_series.ewm(span=p, adjust=False).mean().iloc[-1] for p in periods]

        groups = {
            "micro": (self.sub_1, self.sub_2, self.sub_3, [0.5, 0.3, 0.2]),
            "structure": (self.sub_4, self.sub_5, self.sub_6, [0.4, 0.4, 0.2]),
            "macro": (self.sub_7, self.sub_8, self.sub_9, [0.33, 0.33, 0.34])
        }

        all_emas = []
        for name, (g1, g2, g3, weights) in groups.items():
            e1, e2, e3 = get_emas(g1), get_emas(g2), get_emas(g3)
            all_emas.extend(e1 + e2 + e3)
            int_g = self.compute_group_integrity(e1) * weights[0] + \
                    self.compute_group_integrity(e2) * weights[1] + \
                    self.compute_group_integrity(e3) * weights[2]
            metrics[f"integrity_{name if name != 'structure' else 'operative'}"] = int_g
            div = self.compute_chromatic_divergence(e1 + e2 + e3, atr)
            metrics[f"{name if name != 'structure' else 'operative'}_spectrum"] = int_g * div

        metrics["bias_level"] = get_emas(self.sub_10)[0]
        return metrics

class PhysicsEngine:
    """
    [LEVEL 48-DOCTORAL] UNIFIED PHYSICS & THERMODYNAMICS ENGINE
    Consolidated source of truth for Institutional Force, Energy, and Entropy.
    """
    def __init__(self):
        self.eps = getattr(settings, "PHYSICS_MIN_ATR", 1e-9)
        self.layers = {
            "micro": settings.LAYER_MICRO_PERIODS,
            "structure": settings.LAYER_OPERATIVE_PERIODS,
            "macro": settings.LAYER_MACRO_PERIODS
        }
        self.bias_period = settings.LAYER_BIAS_PERIOD
        self.validator = KineticValidator(self.eps)
        self.spectral = SpectralMechanics(self.eps)
        # Deferred import to avoid circular dependency
        from nexus_core.adaptation_engine import adaptation_engine
        self.adaptation_engine = adaptation_engine

    def compute_vectorized_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized core for Training and Deep Analysis."""
        res = pd.DataFrame(index=df.index)
        close = df["close"]
        open_p = df["open"]
        high = df["high"]
        low = df["low"]
        
        # 1. SCALE INVARIANCE (ATR & RVOL)
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        # [HARDENING] Initial periods must be 1 to prevent NaN in rolling window during early training rows
        atr = tr.rolling(14, min_periods=1).mean()
        atr = atr.replace(0, self.eps).fillna(self.eps)
        
        if "volume" in df.columns:
            vol_mean = df["volume"].rolling(20, min_periods=1).mean()
            rvol = df["volume"] / (vol_mean + self.eps)
        else:
            rvol = pd.Series(1.0, index=df.index)

        # 2. NEWTONIAN MECHANICS (Force & Energy)
        c_micro = close.ewm(span=self.layers["micro"][0], adjust=False).mean()
        c_macro = close.ewm(span=self.layers["macro"][0], adjust=False).mean()
        
        raw_force = (abs(close - c_micro) / atr) * rvol
        raw_energy = (abs(close - c_macro) / atr) * rvol
        
        # [HARDENING] Institutional Fakeout Defense
        # If force is high but volume is low, it's a gap/void, not a drive.
        mask_fakeout = (raw_force > 2.0) & (rvol < 0.8)
        raw_force[mask_fakeout] = 0.0

        res["physics_force"] = np.log1p(raw_force)
        res["physics_energy"] = np.log1p(raw_energy)
        res["feat_force"] = res["physics_force"] # Aligned with NEURAL_FEATURE_NAMES

        # 3. THERMODYNAMICS (Entropy & Viscosity)
        res["physics_entropy"] = (close.rolling(10, min_periods=1).std() / atr).clip(0, 1)
        body = (close - open_p).abs()
        range_s = (high - low) + self.eps
        res["physics_viscosity"] = (1.0 - (body / range_s)).clip(0, 1)

        # 4. VECTORIZED ACCELERATION
        res["physics_accel"] = res["physics_force"].diff().fillna(0.0)
        res["physics_efficiency"] = body / ((df["volume"] if "volume" in df.columns else 1.0) * atr + self.eps)

        # 5. TRAINING TARGETS (Retroactive Labeling)
        # Identifying impulses for the Neural Net to learn specifically
        is_impulse = (body > 1.5 * atr) & (raw_force > 2.0)
        res["target_kinetic_state_impulse"] = is_impulse.astype(float)
        
        # Verification labels (Retroactive lookahead)
        limit_bull = low + 0.5 * body
        limit_bear = high - 0.5 * body
        
        # Check if future candles (up to 3) defend the 50% level
        defend_bull = (close.shift(-1) >= limit_bull) & (close.shift(-2) >= limit_bull) & (close.shift(-3) >= limit_bull)
        defend_bear = (close.shift(-1) <= limit_bear) & (close.shift(-2) <= limit_bear) & (close.shift(-3) <= limit_bear)
        
        bull_impulse = is_impulse & (close > open_p)
        bear_impulse = is_impulse & (close < open_p)
        
        res["target_kinetic_state_confirmed"] = ((bull_impulse & defend_bull) | (bear_impulse & defend_bear)).astype(float)
        res["target_kinetic_state_failed"] = ((bull_impulse & ~defend_bull) | (bear_impulse & ~defend_bear)).astype(float)
        res["target_kinetic_state_neutral"] = (1.0 - res["target_kinetic_state_impulse"]).clip(0, 1)
        
        # 6. [M2-M3 CORRELATION] Potential Energy & Viscosity Modifier
        # These are placeholders that will be populated by the StructureEngine
        # when FVG zones are detected. We initialize them here for schema consistency.
        res["potential_energy"] = pd.Series(0.0, index=df.index)
        res["viscosity_modifier"] = pd.Series(1.0, index=df.index)  # 1.0 = no modification
        res["acceleration_quality"] = pd.Series(1.0, index=df.index)  # 1.0 = healthy, <1.0 = artificial
        
        # Artificial Acceleration Detection (High Force + Low Volume)
        if "volume" in df.columns:
            vol_mean = df["volume"].rolling(20, min_periods=1).mean()
            rvol = df["volume"] / (vol_mean + self.eps)
            # If force is high but volume is low, acceleration is "artificial"
            artificial_mask = (res["physics_force"] > 1.5) & (rvol < 0.7)
            res.loc[artificial_mask, "acceleration_quality"] = 0.5

        return res.fillna(0.0)

    def compute_live_state(self, df: pd.DataFrame) -> Dict[str, float]:
        """Real-time iterative engine for high-frequency execution."""
        if df.empty: return {}
        
        # 1. Base Metrics
        v_physics = self.compute_vectorized_physics(df).iloc[-1].to_dict()
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        if atr <= 0: atr = self.eps
        
        # 2. Spectral Integration
        spec = self.spectral.analyze_spectrum(df["close"], atr=atr)
        metrics = {**v_physics, **spec}
        
        # 3. Absorption State Machine
        vol_scalar = metrics.get("volatility_context", 1.0)
        limit_pct = self.adaptation_engine.get_retracement_limit(vol_scalar)
        window = self.adaptation_engine.get_absorption_window(vol_scalar)
        
        abs_res = self.validator.check_absorption_state(df, limit_pct=limit_pct, window=window)
        state_map = {"NEUTRAL": 0.0, "IMPULSE": 1.0, "MONITORING": 2.0, "CONFIRMED": 3.0, "FAILED": 4.0}
        metrics["absorption_state"] = state_map.get(abs_res["state"], 0.0)
        metrics["absorption_progress"] = float(abs_res["progress"])
        
        return metrics

# Singleton
physics_engine = PhysicsEngine()
