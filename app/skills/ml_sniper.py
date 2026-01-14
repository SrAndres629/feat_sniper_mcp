import logging
import numpy as np
from typing import Dict, Any, List
from app.skills import indicators, market

logger = logging.getLogger("MT5_Bridge.Skills.ML")

class MLSniper:
    """
    ML-Lite Setup Classifier 2.0.
    Uses pattern confluence to score trading setups.
    """

    @staticmethod
    async def score_setup(symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Calculates a 'Confidence Score' using multiple indicator convergences.
        Simulates supervised classification (RandomForest logic).
        """
        # 1. Fetch multi-TF data
        h1_vola = await market.get_volatility_metrics(symbol, "H1")
        m15_rsi = await indicators.calculate_rsi(symbol, "M15", 14)

        # Fetch 3 candles for acceleration calculation (t, t-1, t-2)
        m5_candles_data = await market.get_candles(symbol, "M5", 3)
        m5_candles = m5_candles_data.get("candles", [])
        m5_price = m5_candles[-1] if m5_candles else None

        # Calculate Physics (Velocity & Acceleration)
        velocity = 0.0
        acceleration = 0.0
        trap_detected = False
        trap_type = "NONE"

        if len(m5_candles) >= 3:
            p0 = m5_candles[-3]['close']
            p1 = m5_candles[-2]['close']
            p2 = m5_candles[-1]['close']

            v1 = p1 - p0
            v2 = p2 - p1

            velocity = v2
            acceleration = v2 - v1

            # Retail Trap Logic: Divergence between Velocity and Acceleration
            # Bullish Trap: Price Rising (V > 0) but Decelerating (A < 0) -> Exhaustion?
            if velocity > 0 and acceleration < 0:
                trap_detected = True
                trap_type = "BULLISH_EXHAUSTION"
            # Bearish Trap: Price Falling (V < 0) but Accelerating upwards (slowing down fall) (A > 0)
            elif velocity < 0 and acceleration > 0:
                trap_detected = True
                trap_type = "BEARISH_EXHAUSTION"

        # 2. Heuristic Features (Proxies for ML features)
        features = {
            "vola_status": h1_vola["volatility_status"] == "NORMAL",
            "rsi_oversold": m15_rsi < 35,
            "rsi_overbought": m15_rsi > 65,
            "is_liquid": h1_vola["spread_points"] < 20,
            "trap_detected": trap_detected,
            "acceleration": round(acceleration, 5)
        }

        # 3. Model Weighting (Simulated logic)
        score = 0
        if features["vola_status"]: score += 0.3
        if features["rsi_oversold"] or features["rsi_overbought"]: score += 0.4
        if features["is_liquid"]: score += 0.3

        # Alpha Refinement: Trap Logic Integration
        # If we have a Reversal Signal (Overbought/Oversold) AND a Trap (Exhaustion), this is a strong signal.
        if trap_detected:
            if features["rsi_oversold"] and trap_type == "BEARISH_EXHAUSTION":
                # Price falling but slowing down + Oversold -> Strong Buy
                score += 0.15
            elif features["rsi_overbought"] and trap_type == "BULLISH_EXHAUSTION":
                # Price rising but slowing down + Overbought -> Strong Sell
                score += 0.15
            else:
                 # Trap without signal might just be noise or choppy market
                 pass

        # 4. Outlier Detection (Anomalies)
        anomaly = False
        if h1_vola["volatility_status"] == "EXTREME":
            anomaly = True
            score = score * 0.5 # Penalty for extreme noise

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "confidence_score": round(score, 2),
            "setup_quality": "HIGH" if score > 0.7 else ("MEDIUM" if score > 0.4 else "LOW"),
            "is_anomaly": anomaly,
            "features_snapshot": features,
            "physics": {
                "velocity": round(velocity, 5),
                "acceleration": round(acceleration, 5),
                "trap_type": trap_type
            }
        }

ml_sniper = MLSniper()
