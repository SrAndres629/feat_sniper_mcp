import logging
import pandas as pd
from typing import Dict, Any, List
from app.core.config import settings

from .patterns import MAE_Pattern_Recognizer
from .fractals import identify_fractals
from .zones import detect_zones
from .transitions import detect_structural_shifts
from .imbalances import detect_imbalances
from .liquidity import detect_liquidity_pools
from .order_blocks import detect_order_blocks
from .trap_detector import calculate_trap_score, get_trap_report
from .ema_layers import four_layer_ema
from .reports import StructureReporter

logger = logging.getLogger("feat.structure")

class StructureEngine:
    """
    [v5.0 - SMC] Institutional Structure quantification with Trap Awareness.
    Philosophy: The more 'perfect' a setup, the more likely it's a trap.
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
    
    def detect_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_zones(df)

    def compute_feat_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.detect_structural_shifts(df)
        df = self.detect_imbalances(df)
        df = self.detect_zones(df)
        df = detect_liquidity_pools(df)
        df = detect_order_blocks(df)
        df = calculate_trap_score(df)  # [PREDATORY AWARENESS]
        mae = self.mae_recognizer.detect_mae_pattern(df)

        last_row = df.iloc[-1]
        
        # 1. Structural Force (BOS + CHOCH)
        f_score = last_row.get("bos_strength", 0.0)
        if last_row["choch_bull"] or last_row["choch_bear"]: f_score += 0.5
        
        # 2. Institutional Footprint (Order Blocks + Mitigation)
        ob_score = 0.0
        if (last_row["ob_bull"] or last_row["ob_bear"]):
            ob_score = 1.0 if not last_row["is_mitigated"] else 0.3
            
        # 3. Institutional Gravity (FVG + Liquidity Pools)
        liq_gravity = 0.5 if (last_row["is_eqh"] or last_row["is_eql"]) else 0.0
        e_score = (1.0 if (last_row["fvg_bull"] or last_row["fvg_bear"]) else 0.2) + liq_gravity + ob_score

        # Zone Fragility Logic
        if last_row["test_count"] >= 3:
            f_score += 0.3 

        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        curr_range = df["high"].iloc[-1] - df["low"].iloc[-1]
        a_score = min(1.0, curr_range / (atr * 2 + 1e-9))

        t_score = 0.5
        w = self.config["weights"]
        raw_feat = (w["F"] * f_score + w["E"] * e_score + w["A"] * a_score + w["T"] * t_score)
        
        # [PREDATORY AWARENESS] Apply Trap Penalty
        trap_score = last_row.get("trap_score", 0.0)
        trap_penalty = 1.0 - (trap_score * 0.5)  # Reduce by up to 50%
        feat_val = raw_feat * trap_penalty

        res = pd.DataFrame(index=df.index)
        res["feat_index"] = round(feat_val * 100, 2)
        res["trap_score"] = round(trap_score, 2)
        res["structure_status"] = mae["status"]
        res["is_mae_expansion"] = mae["is_expansion"]
        return res

    def get_structural_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = self.detect_structural_shifts(df)
        df = self.detect_imbalances(df)
        df = self.detect_zones(df)
        df = detect_liquidity_pools(df)
        df = detect_order_blocks(df)
        df = calculate_trap_score(df)
        mae = self.mae_recognizer.detect_mae_pattern(df)
        trap = get_trap_report(df)
        last_row = df.iloc[-1]
        
        return {
            "mae_pattern": {
                "phase": mae["phase"],
                "status": mae["status"],
                "session": last_row.get("session_type", "NONE")
            },
            "institutional": {
                "nearest_ob": "BULL" if last_row["ob_bull"] else "BEAR" if last_row["ob_bear"] else "NONE",
                "is_mitigated": bool(last_row["is_mitigated"]),
                "has_liquidity_pool": bool(last_row["is_eqh"] or last_row["is_eql"]),
                "has_fvg": bool(last_row["fvg_bull"] or last_row["fvg_bear"]),
                "hunter_mode": bool(last_row.get("hunter_mode", False))
            },
            "trap_analysis": trap,  # [PREDATORY AWARENESS]
            "zones": {
                "nearest_zone": last_row["zone_type"],
                "test_count": int(last_row["test_count"])
            }
        }

    def get_structural_health(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Legacy placeholder for stat validator
        return {"health_score": 0.5, "status": "STABLE"}

    def get_structural_score(self, df: pd.DataFrame) -> float:
        report = self.get_structural_report(df)
        score = (
            report['health']['health_score'] * 0.4 +
            report['zones']['zone_strength'] * 0.3 +
            report['mae_pattern']['is_expansion'] * 0.2 +
            report['health']['trend_alignment'] * 0.1
        ) * 100
        return round(float(score), 2)
