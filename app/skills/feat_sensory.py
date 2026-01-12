"""
FEAT SENSORY STACK (MSS-5) - Protocol v2.5
==========================================
The "Nervous System" of FEAT NEXUS. Converts raw market physics into 
Neural State Tensors for ML consumption.

Tensors:
1. Inertia (Trend persistence)
2. Entropy Coeff (Volatility regime)
3. Harmonic Phase (Cycle position)
4. Kinetic Tension (Momentum exhaustion)
5. Mass Flow (Institutional volume)
6. Acceptance Ratio (Structural solidness)
7. Wick Stress (Rejection pressure)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from app.skills.market_physics import market_physics
from app.skills.indicators import get_technical_indicator, IndicatorRequest

logger = logging.getLogger("FEAT.Sensory")

class SensoryStack:
    def __init__(self):
        self.resonance_threshold = 0.65

    def calculate_mss5_tensors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Synthesizes raw data into 8 Neural Tensors.
        """
        if df.empty or len(df) < 50:
            return self._empty_response()

        try:
            # 1. PHYSICS LAYER integration
            pvp_feat = market_physics.calculate_pvp_feat(df)
            cvd_feat = market_physics.calculate_cvd_metrics(df)
            
            # 2. CALC TENSORS
            
            # T1: Inertia (EMA 50/200 Relationship)
            ema50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema200 = df['close'].ewm(span=200).mean().iloc[-1]
            inertia = 1.0 if ema50 > ema200 else 0.0
            
            # T2: Entropy Coeff (ADX 14 - Trend Strength)
            # Higher ADX = Lower Entropy (Stronger Trend)
            adx_data = get_technical_indicator(df, IndicatorRequest(symbol="SYM", indicator="ADX"))
            entropy_coeff = adx_data.get("value", 0) / 100.0
            
            # T3: Harmonic Phase (MACD Cross Status)
            macd_data = get_technical_indicator(df, IndicatorRequest(symbol="SYM", indicator="MACD"))
            harmonic_phase = 1.0 if macd_data.get("histogram", 0) > 0 else -1.0
            
            # T4: Kinetic Tension (RSI 9 - Exhaustion)
            rsi_data = get_technical_indicator(df, IndicatorRequest(symbol="SYM", indicator="RSI", period=9))
            rsi = rsi_data.get("value", 50)
            kinetic_tension = (rsi - 50) / 50.0 # -1 to 1
            
            # T5: Institutional Mass Flow (CVD Acceleration normalized)
            mass_flow = np.tanh(cvd_feat["cvd_acceleration"] / 1000.0) # Normalized -1 to 1
            
            # T6: Volatility Regime (ATR / Z-Score Hybrid)
            atr_data = get_technical_indicator(df, IndicatorRequest(symbol="SYM", indicator="ATR"))
            volatility_norm = min(1.0, atr_data.get("value", 0) / (df['close'].iloc[-1] * 0.01))
            
            # T7: Acceptance Ratio (Body / Range)
            body = (df['close'] - df['open']).abs()
            candle_range = (df['high'] - df['low'])
            acc_ratio = (body / (candle_range + 1e-9)).iloc[-1]
            
            # T8: Wick Stress (Rejection intensity)
            wick_stress = 1.0 - acc_ratio
            
            # 3. RESONANCE SYNTHESIS
            # Resonance occurs when multiple tensors align in the same direction
            directional_alignment = np.sign(kinetic_tension) == np.sign(mass_flow) == np.sign(harmonic_phase)
            resonance_score = 0.5
            if directional_alignment:
                resonance_score = 0.8 + (0.2 * entropy_coeff)
            
            # Apply PVP Context to resonance
            # If price is at POC, resonance is neutralized (low interest)
            poc_dist = abs(pvp_feat["z_score"])
            if poc_dist < 0.5:
                resonance_score *= 0.7
            
            return {
                "tensors": {
                    "inertia": float(inertia),
                    "entropy_coeff": float(entropy_coeff),
                    "harmonic_phase": float(harmonic_phase),
                    "kinetic_tension": float(kinetic_tension),
                    "mass_flow": float(mass_flow),
                    "volatility_norm": float(volatility_norm),
                    "acceptance_ratio": float(acc_ratio),
                    "wick_stress": float(wick_stress)
                },
                "resonance_score": float(resonance_score),
                "validation": "NOMINAL" if resonance_score > self.resonance_threshold else "LOW_CONFIDENCE",
                "physics_context": pvp_feat
            }

        except Exception as e:
            logger.error(f"Error calculating MSS-5 Tensors: {e}")
            return self._empty_response()

    def _empty_response(self):
        return {
            "tensors": {},
            "resonance_score": 0.0,
            "validation": "ERROR",
            "physics_context": {}
        }

# Singleton
sensory_stack = SensoryStack()

def calculate_mss5_tensors(df: pd.DataFrame) -> Dict[str, Any]:
    return sensory_stack.calculate_mss5_tensors(df)
