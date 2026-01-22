import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from app.core.config import settings

from .patterns import MAE_Pattern_Recognizer
from .fractals import identify_fractals
from .transitions import detect_structural_shifts
from .imbalances import detect_imbalances
from .liquidity import detect_liquidity_pools
from .order_blocks import detect_order_blocks
from .trap_detector import calculate_trap_score, get_trap_report
from .critical_points import detect_critical_points
from .shadow_zones import detect_shadow_zones
from .consolidation_zones import detect_consolidation_zones
from .ema_layers import four_layer_ema
from .reports import StructureReporter

logger = logging.getLogger("feat.structure")

class StructureEngine:
    """
    [v5.1 - SMC VECTORIZED] Institutional Structure quantification with Trap Awareness.
    Refactored by the Structure Department for massive backtesting efficiency.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "atr_window": 14,
            "weights": settings.STRUCT_WEIGHTS,
        }
        self.mae_recognizer = MAE_Pattern_Recognizer()
        self.reporter = StructureReporter(self)

    def identify_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        return identify_fractals(df)

    def detect_structural_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_structural_shifts(df)

    def detect_imbalances(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_imbalances(df)

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_order_blocks(df)

    def calculate_confluence_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [THE GOLDEN RULE - v6] 
        Validates zone overlaps with adaptive weights and persistent states.
        """
        if df.empty: 
            df["confluence_score"] = 0.0
            return df
            
        # Weights Definitions
        w_ob = 1.0
        w_fvg = 0.8
        w_breaker = 0.7 
        w_liq = 0.6
        
        # 1. OB Contribution (State Based)
        # 1:Active (Full Weight), 2:Mitigated (Half Weight), 3:Breaker (Specific Weight)
        ob_score = np.zeros(len(df))
        if "ob_bull" in df.columns:
            ob_score += np.where(df["ob_bull"] == 1.0, w_ob, 0.0)
            ob_score += np.where(df["ob_bull"] == 2.0, w_ob * 0.5, 0.0)
        if "ob_bear" in df.columns:
            ob_score += np.where(df["ob_bear"] == 1.0, w_ob, 0.0)
            ob_score += np.where(df["ob_bear"] == 2.0, w_ob * 0.5, 0.0)
            
        # 2. FVG Contribution (Gravity Based)
        fvg_score = df["fvg_gravity"].abs().clip(0, 1) * w_fvg if "fvg_gravity" in df.columns else 0.0
        
        # 3. Breaker Contribution
        breaker_score = np.zeros(len(df))
        if "ob_bull" in df.columns:
             breaker_score += np.where(df["ob_bull"] == 3.0, w_breaker, 0.0)
        if "ob_bear" in df.columns:
             breaker_score += np.where(df["ob_bear"] == 3.0, w_breaker, 0.0)

        # 4. Liquidity Pools
        has_liq = ((df.get("is_eqh", 0) + df.get("is_eql", 0)) > 0).astype(float) * w_liq
        
        # Aggregate
        score = ob_score + fvg_score + breaker_score + has_liq
        
        # Temporal Alignment (H1/H4 Context)
        if "session_weight" in df.columns:
            score = score * df["session_weight"]
            
        df["confluence_score"] = score.round(2)
        return df

    def compute_feat_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [v6.2 DOCTORAL SEAL: M2-M3 FULL CORRELATION]
        Unified sensory output with complete Physics-Geometry coupling.
        Viscosity is now geometrically adjusted based on FVG zones.
        Acceleration Quality (Q_a) = Acceleration × RVOL_Normalized.
        """
        df = self.detect_structural_shifts(df)
        df = self.detect_imbalances(df)  # Populates physics and FVG columns
        df = detect_liquidity_pools(df)
        df = detect_order_blocks(df)
        df = calculate_trap_score(df) 
        
        df = detect_critical_points(df)
        df = detect_shadow_zones(df)
        df = detect_consolidation_zones(df)
        df = self.calculate_confluence_score(df)
        
        # 1. Structural Conviction (F-Score)
        f_score = df.get("bos_strength", pd.Series(0.0, index=df.index)).fillna(0.0)
        
        # 2. Institutional Gravity (G-Score) - From FVG Duality
        gravity_score = df.get("fvg_gravity", pd.Series(0.0, index=df.index)).abs().clip(0, 1.5)
        
        # 3. Propulsion Score (P-Score) - Trend Continuation from FVG Duality
        propulsion_score = df.get("fvg_propulsion", pd.Series(0.0, index=df.index)).abs().clip(0, 1.5)
        
        # 4. Space Quality (S-Score)
        s_score = df.get("range_pos", pd.Series(0.5, index=df.index)).fillna(0.5)

        # ===================================================
        # 5. [M2-M3 SEALED CORRELATION] Geometric Coupling
        # ===================================================
        from nexus_core.physics_engine.engine import physics_engine
        physics_res = physics_engine.compute_vectorized_physics(df)
        
        # Base Physics Metrics
        base_viscosity = physics_res.get("physics_viscosity", pd.Series(1.0, index=df.index))
        base_accel = physics_res.get("physics_accel", pd.Series(0.0, index=df.index))
        
        # RVOL for Acceleration Quality
        if "volume" in df.columns:
            vol_mean = df["volume"].rolling(20, min_periods=1).mean()
            rvol = df["volume"] / (vol_mean + 1e-9)
        else:
            rvol = pd.Series(1.0, index=df.index)
        
        # [SEAL 1] VISCOSITY GEOMETRIC COUPLING
        # Rule: If price is inside a GRAVITY FVG zone, reduce viscosity by 30%
        # This represents the "vacuum" effect of the gap.
        from nexus_core.structure_engine.imbalances import FVG_GRAVITY
        is_in_gravity_zone = (df.get("fvg_type", None) == FVG_GRAVITY)
        
        viscosity_modifier = pd.Series(1.0, index=df.index)
        viscosity_modifier[is_in_gravity_zone] = 0.7  # 30% reduction
        
        # Also reduce viscosity if propulsion is high (runaway gap = zero friction)
        high_propulsion = propulsion_score > 0.5
        viscosity_modifier[high_propulsion] = viscosity_modifier[high_propulsion] * 0.8  # 20% additional
        
        df["viscosity_modifier"] = viscosity_modifier
        
        # [SEAL 2] ACCELERATION QUALITY (Q_a = Accel × RVOL_Normalized)
        # Rule: High acceleration with low volume = artificial movement
        accel_magnitude = base_accel.abs()
        rvol_normalized = (rvol / (rvol.rolling(20, min_periods=1).mean() + 1e-9)).clip(0.1, 3.0)
        
        # Q_a = Normalized product
        acceleration_quality = (accel_magnitude * rvol_normalized).clip(0.0, 2.0)
        
        # Penalty: If accel is high (>0.5) but RVOL is low (<0.7), quality drops
        artificial_mask = (accel_magnitude > 0.5) & (rvol < 0.7)
        acceleration_quality[artificial_mask] = acceleration_quality[artificial_mask] * 0.3
        
        # Normalize to [0, 1] for neural consumption
        df["acceleration_quality"] = (acceleration_quality / 2.0).clip(0, 1)
        
        # [SEAL 3] POTENTIAL ENERGY
        fvg_mid = (df.get("fvg_bull_top", 0.0) + df.get("fvg_bull_bottom", 0.0)) / 2
        distance_to_gap = (df["close"] - fvg_mid).abs() + 1e-9
        df["potential_energy"] = (gravity_score / distance_to_gap).clip(0, 10)

        # ===================================================
        # 6. Final Aggregation (TCN Channels 7-14)
        # ===================================================
        feat_val = (f_score * 0.25 + gravity_score * 0.2 + propulsion_score * 0.2 + s_score * 0.15)
        
        # Add quality metrics contribution (Channels 13-14)
        feat_val += df["acceleration_quality"] * 0.1
        feat_val += (1.0 - df["viscosity_modifier"]) * 0.1  # Lower viscosity = higher feat
        
        # Trap Penalty
        trap_penalty = 1.0 - (df.get("trap_score", 0.0) * 0.5)
        feat_val = feat_val * trap_penalty
        
        df["feat_index"] = (feat_val * 100).round(2)
        
        # Map additional fields for AlphaTensor
        if "structural_feat_index" not in df.columns:
             df["structural_feat_index"] = df["feat_index"] / 100.0
             
        return df

    def get_structural_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detailed report for the current candle."""
        df = self.compute_feat_index(df) # Ensure all features are calculated
        trap = get_trap_report(df)
        last_row = df.iloc[-1]
        
        return {
            "mae_pattern": {
                "status": last_row.get("structure_status", "UNKNOWN"),
                "is_expansion": bool(last_row.get("is_mae_expansion", False))
            },
            "institutional": {
                "nearest_ob": "BULL" if last_row.get("ob_bull") else "BEAR" if last_row.get("ob_bear") else "NONE",
                "is_mitigated": bool(last_row.get("is_mitigated", False)),
                "has_liquidity_pool": bool(last_row.get("is_eqh", False) or last_row.get("is_eql", False)),
                "has_fvg": bool(last_row.get("fvg_bull", False) or last_row.get("fvg_bear", False))
            },
            "space_analysis": {
                "confluence_score": float(last_row.get("confluence_score", 0.0)),
                "feat_index": float(last_row.get("feat_index", 0.0))
            },
            "trap_analysis": trap
        }

    def get_structural_health(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"health_score": 0.5, "status": "STABLE"}

    def get_structural_score(self, df: pd.DataFrame) -> float:
        res = self.compute_feat_index(df)
        return float(res["feat_index"].iloc[-1])

# Singleton
structure_engine = StructureEngine()
