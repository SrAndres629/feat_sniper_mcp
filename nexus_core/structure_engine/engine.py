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
        [THE GOLDEN RULE - VECTORIZED] 
        Validates zone overlaps with adaptive weights across entire history.
        """
        if df.empty: 
            df["confluence_score"] = 0.0
            return df
            
        # Weights Definitions
        w_ob = 1.0
        w_fvg = 0.8
        w_breaker = 0.7 
        w_liq = 0.6
        
        # Vectorized Presence Flags
        # [FIX] Arithmetic Logic (Safe for Float/NaN)
        # We sum the columns (fillna 0) and check if > 0.
        def _any(c1, c2):
            # Ensure we default to a Series of 0.0 if column missing
            v1 = df[c1].fillna(0) if c1 in df.columns else pd.Series(0.0, index=df.index)
            v2 = df[c2].fillna(0) if c2 in df.columns else pd.Series(0, index=df.index)
            return ((v1 + v2) > 0).astype(float)
            
        has_ob = _any("ob_bull", "ob_bear")
        has_fvg = _any("fvg_bull", "fvg_bear")
        has_breaker = _any("breaker_bull", "breaker_bear")
        has_liq = _any("is_eqh", "is_eql")
        
        # Mitigation Decay
        ob_weight = np.where(df.get("is_mitigated", False), w_ob * 0.5, w_ob)
        fvg_weight = np.where(df.get("fvg_mitigated", False), w_fvg * 0.5, w_fvg)
        
        # Aggregate Score
        score = (has_ob * ob_weight) + (has_fvg * fvg_weight) + (has_breaker * w_breaker) + (has_liq * w_liq)
        
        # Temporal Alignment
        if "major_h" in df.columns:
            # [FIX] Arithmetic Logic
            align_val = (df["major_h"].fillna(0) + df["major_l"].fillna(0))
            alignment = (align_val > 0).astype(float).rolling(50).max().fillna(0)
            score = score * (1.0 + (alignment * 0.5))
            
        df["confluence_score"] = score.round(2)
        return df

    def compute_feat_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [CHRONOS VECTORIZED FEAT INDEX]
        Calculates the FEAT Index relative to ATR and Structural Force.
        """
        df = self.detect_structural_shifts(df)
        df = self.detect_imbalances(df)
        df = detect_liquidity_pools(df)
        df = detect_order_blocks(df)
        df = calculate_trap_score(df) 
        
        df = detect_critical_points(df)
        df = detect_shadow_zones(df)
        df = detect_consolidation_zones(df)
        df = self.calculate_confluence_score(df)
        
        
        # [FIX] Helper for Safe Boolean Series Retrieval
        def _get_bool(col_name):
            if col_name in df.columns:
                return df[col_name].fillna(False).astype(bool)
            return pd.Series(False, index=df.index)

        # 1. Structural Force (Vectorized)
        f_score = df.get("bos_strength", pd.Series(0.0, index=df.index)).fillna(0.0)
        
        # [FIX] Arithmetic Logic for CHoCH
        c_bull = _get_bool("choch_bull").astype(int)
        c_bear = _get_bool("choch_bear").astype(int)
        f_score += np.where((c_bull + c_bear) > 0, 0.5, 0.0)
        
        # 2. Institutional Efficiency (E-Score)
        # Safe bitwise OR on Series
        has_ob = _get_bool("ob_bull") | _get_bool("ob_bear")
        has_fvg = _get_bool("fvg_bull") | _get_bool("fvg_bear")
        
        ob_presence = np.where(has_ob, 1.0, 0.0)
        fvg_presence = np.where(has_fvg, 1.0, 0.2)
        e_score = fvg_presence + ob_presence
        
        # 3. ATR-Normalized Range (A-Score)
        atr = (df["high"] - df["low"]).rolling(14).mean().ffill()
        curr_range = df["high"] - df["low"]
        a_score = (curr_range / (atr * 2 + 1e-9)).clip(0, 1)

        # 4. Final Aggregation
        w = self.config["weights"]
        
        # [CHRONOS] Session-based T-Score (Liquidity Time)
        times = df.index
        # Get hour in UTC (or adjusted for exchange time)
        if hasattr(times, 'hour'):
             hours = times.hour
        else:
             # Fallback if index not datetime (should be fixed by vectorized_tensor but defensive here)
             hours = pd.Series(0, index=df.index)
        
        # Scoring: London (7-11 UTC) and NY (12-16 UTC) are 1.0. Asian (23-06 UTC) is 0.5. Weekend/Dead is 0.0.
        t_score = pd.Series(0.2, index=df.index) # Base
        
        if hasattr(times, 'hour'):
            t_score[((hours >= 7) & (hours <= 11)) | ((hours >= 12) & (hours <= 16))] = 1.0 # High Liquidity
            t_score[((hours >= 0) & (hours <= 6))] = 0.5 # Asian
        
        # [FEAT UPDATE] Calculate Space Quality (E) using EMA Layers
        # Optimal Space is when Price pulls back to Structure (dist ~ 0)
        
        if "dist_structure" in df.columns:
            d_struct = df["dist_structure"].abs()
            # Reward being close to Structure (Lower distance = Higher Quality for entry)
            # Scaling: 1.0 at d=0, decaying as price moves away.
            space_quality = 1.0 / (1.0 + d_struct)
        else:
             # If no structure column, space quality is zero (No identifiable floor)
             space_quality = 0.0
            
        raw_feat = (w["F"] * f_score + w["E"] * space_quality + w["A"] * a_score + w["T"] * t_score)
        raw_feat += df["confluence_score"] * 0.2
        
        # 5. Trap Penalty
        trap_penalty = 1.0 - (df.get("trap_score", 0.0) * 0.5)
        feat_val = raw_feat * trap_penalty
        
        # Pattern Metadata (MAE)
        mae = self.mae_recognizer.detect_mae_pattern(df)
        
        res = pd.DataFrame(index=df.index)
        res["feat_index"] = (feat_val * 100).round(2)
        res["trap_score"] = df.get("trap_score", 0.0).round(2)
        res["confluence_score"] = df["confluence_score"].round(2)
        res["structure_status"] = mae["status"]
        res["is_mae_expansion"] = mae["is_expansion"]
        
        # Meta-tensors for Alpha Orchestrator
        res["session_weight"] = df.get("session_weight", 1.0)
        res["struct_displacement_z"] = df.get("struct_displacement_z", 0.0)
        return res

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
