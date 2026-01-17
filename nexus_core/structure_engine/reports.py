from typing import Dict, Any, List
import pandas as pd
from .models import StructuralConfidence

class StructureReporter:
    def __init__(self, engine):
        self.engine = engine

    def get_structural_narrative(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = self.engine.detect_structural_shifts(df)
        last_row = df.iloc[-1]
        bos_level = 0.0
        bos_type = "NONE"

        if last_row["bos_bull"]:
            bos_level = last_row["last_h_fractal"]
            bos_type = "BULLISH_BOS"
        elif last_row["bos_bear"]:
            bos_level = last_row["last_l_fractal"]
            bos_type = "BEARISH_BOS"
        elif last_row["choch_bull"]:
            bos_level = last_row["last_h_fractal"]
            bos_type = "BULLISH_CHOCH"
        elif last_row["choch_bear"]:
            bos_level = last_row["last_l_fractal"]
            bos_type = "BEARISH_CHOCH"

        return {
            "last_bos": float(bos_level),
            "type": bos_type,
            "status": self.engine.mae_recognizer.detect_mae_pattern(df)["status"],
        }

    def get_zone_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = self.engine.detect_zones(df)
        last_row = df.iloc[-1]
        if last_row["zone_type"] == "NONE":
            return {"nearest_zone": "NONE", "distance_to_zone": 0.0, "zone_strength": 0.0, "is_in_zone": False}
        
        current_price = last_row["close"]
        zone_high = last_row["zone_high"]
        zone_low = last_row["zone_low"]
        
        if last_row["zone_type"] == "SUPPLY":
            distance = zone_high - current_price
        else:
            distance = current_price - zone_low
        
        is_in_zone = current_price <= zone_high and current_price >= zone_low
        return {
            "nearest_zone": last_row["zone_type"],
            "distance_to_zone": float(distance),
            "zone_strength": float(last_row["zone_strength"]),
            "is_in_zone": is_in_zone
        }

    def get_structural_summary(self, df: pd.DataFrame) -> str:
        report = self.engine.get_structural_report(df)
        summary = f"Structural Report:\n"
        summary += f"- MAE Phase: {report['mae_pattern']['phase']} ({report['mae_pattern']['status']})\n"
        summary += f"- Health Score: {report['health']['health_score']} ({report['health']['status']})\n"
        summary += f"- Nearest Zone: {report['zones']['nearest_zone']} (Strength: {report['zones']['zone_strength']})\n"
        summary += f"- Trend Alignment: {report['health']['trend_alignment']}\n"
        return summary

    def get_structural_risk(self, df: pd.DataFrame) -> float:
        report = self.engine.get_structural_report(df)
        mae_volatility = 0.0
        phase = report['mae_pattern']['phase']
        if phase == "EXPANSION": mae_volatility = 0.8
        elif phase in ["ACCUMULATION", "DISTRIBUTION"]: mae_volatility = 0.5
        elif phase == "CONTRACTION": mae_volatility = 0.2
        
        zone_proximity = 1.0 - report['zones']['zone_strength']
        trend_risk = 1.0 - report['health']['trend_alignment']
        risk_score = (mae_volatility + zone_proximity + trend_risk) / 3
        return round(float(risk_score * 100), 2)

    def get_structural_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        report = self.engine.get_structural_report(df)
        opportunities = []
        hp = report['health']
        zp = report['zones']
        mp = report['mae_pattern']
        
        if mp['phase'] == "ACCUMULATION" and zp['nearest_zone'] == "DEMAND":
            opportunities.append({
                "type": "BUY", "zone": "DEMAND", "confidence": round(hp['health_score'] * 0.8, 2),
                "risk_reward_ratio": 1.0 / (1.0 - zp['zone_strength'] + 1e-9),
                "reason": "Price in Demand zone after Accumulation"
            })
        if mp['phase'] == "DISTRIBUTION" and zp['nearest_zone'] == "SUPPLY":
            opportunities.append({
                "type": "SELL", "zone": "SUPPLY", "confidence": round(hp['health_score'] * 0.8, 2),
                "risk_reward_ratio": 1.0 / (1.0 - zp['zone_strength'] + 1e-9),
                "reason": "Price in Supply zone after Distribution"
            })
        return opportunities
