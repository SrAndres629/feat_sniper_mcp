import logging
import pandas as pd
from typing import Dict, Any, List
from app.core.config import settings

from .patterns import MAE_Pattern_Recognizer
from .fractals import identify_fractals
from .zones import detect_zones
from .transitions import detect_structural_shifts
from .ema_layers import four_layer_ema
from .reports import StructureReporter

logger = logging.getLogger("feat.structure")

class StructureEngine:
    """
    Institutional Structure quantification: Fractals, BOS, CHOCH.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "atr_window": 14,
            "weights": {"F": 0.35, "E": 0.25, "A": 0.20, "T": 0.20},
        }
        self.mae_recognizer = MAE_Pattern_Recognizer()
        self.reporter = StructureReporter(self)

    def identify_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        return identify_fractals(df)

    def detect_structural_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_structural_shifts(df)

    def get_structural_narrative(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.reporter.get_structural_narrative(df)

    def get_zone_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        return self.reporter.get_zone_status(df)

    def get_structural_summary(self, df: pd.DataFrame) -> str:
        return self.reporter.get_structural_summary(df)

    def get_structural_risk(self, df: pd.DataFrame) -> float:
        return self.reporter.get_structural_risk(df)

    def get_structural_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        return self.reporter.get_structural_opportunities(df)

    def compute_feat_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.detect_structural_shifts(df)
        mae = self.mae_recognizer.detect_mae_pattern(df)

        f_score = 0.0
        if mae["phase"] == "EXPANSION": f_score += 0.4
        if df["bos_bull"].iloc[-1] or df["bos_bear"].iloc[-1]: f_score += 0.3
        if df["choch_bull"].iloc[-1] or df["choch_bear"].iloc[-1]: f_score += 0.3

        fvg_bull = (df["low"] > df["high"].shift(2)).iloc[-1]
        fvg_bear = (df["high"] < df["low"].shift(2)).iloc[-1]
        e_score = 0.5 if (fvg_bull or fvg_bear) else 0.2

        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        curr_range = df["high"].iloc[-1] - df["low"].iloc[-1]
        a_score = min(1.0, curr_range / (atr * 2 + 1e-9))

        t_score = 0.5
        w = self.config["weights"]
        feat_val = (w["F"] * f_score + w["E"] * e_score + w["A"] * a_score + w["T"] * t_score)

        res = pd.DataFrame(index=df.index)
        res["feat_index"] = round(feat_val * 100, 2)
        res["structure_status"] = mae["status"]
        res["is_mae_expansion"] = mae["is_expansion"]
        return res

    def detect_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        return detect_zones(df)

    def get_structural_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        mae = self.mae_recognizer.detect_mae_pattern(df)
        zones = self.detect_zones(df)
        health = self.get_structural_health(df)
        
        return {
            "mae_pattern": {
                "phase": mae["phase"],
                "is_expansion": mae["is_expansion"],
                "status": mae["status"]
            },
            "zones": {
                "nearest_zone": zones.iloc[-1]["zone_type"],
                "distance_to_zone": float(zones.iloc[-1]["zone_high"] - zones.iloc[-1]["zone_low"]),
                "zone_strength": float(zones.iloc[-1]["zone_strength"])
            },
            "health": health
        }

    def get_structural_health(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = self.detect_structural_shifts(df)
        df = self.detect_zones(df)
        last_row = df.iloc[-1]
        
        bos_strength = 0.0
        if last_row["bos_bull"] or last_row["bos_bear"]: bos_strength = 0.8
        elif last_row["choch_bull"] or last_row["choch_bear"]: bos_strength = 0.5
        
        zone_strength = last_row["zone_strength"]
        
        trend_alignment = 0.5
        if last_row["fractal_high"] and last_row["fractal_low"]:
            if last_row["close"] > last_row["last_h_fractal"]: trend_alignment = 0.8
            elif last_row["close"] < last_row["last_l_fractal"]: trend_alignment = 0.8
        
        health_score = (bos_strength + zone_strength + trend_alignment) / 3
        return {
            "health_score": round(health_score, 2),
            "bos_strength": round(bos_strength, 2),
            "zone_strength": round(zone_strength, 2),
            "trend_alignment": round(trend_alignment, 2),
            "status": "HEALTHY" if health_score > 0.6 else "NEUTRAL" if health_score > 0.3 else "WEAK"
        }

    def get_structural_score(self, df: pd.DataFrame) -> float:
        report = self.get_structural_report(df)
        score = (
            report['health']['health_score'] * 0.4 +
            report['zones']['zone_strength'] * 0.3 +
            report['mae_pattern']['is_expansion'] * 0.2 +
            report['health']['trend_alignment'] * 0.1
        ) * 100
        return round(float(score), 2)
