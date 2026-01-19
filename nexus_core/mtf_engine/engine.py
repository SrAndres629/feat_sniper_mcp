import logging
import pandas as pd
from typing import Dict, Any, List
from .models import Timeframe, TimeframeScore, MTFCompositeScore, get_tf_weight
from .scoring import analyze_hydrodynamics
from app.skills.market_physics import market_physics
from app.core.config import settings

logger = logging.getLogger("feat.mtf")

class FractalMTFEngine:
    """
    Decodifica la "Intención Institucional" analizando OFI e Impacto en 8 TFs.
    """
    def __init__(self):
        logger.info("[MTF] Fractal Microstructure Engine initialized")

    async def analyze_all_timeframes(
        self,
        candles_by_tf: Dict[str, pd.DataFrame],
        current_price: float
    ) -> MTFCompositeScore:
        result = MTFCompositeScore()
        result.reasoning = []
        directions = []
        tf_scores = {}
        flow_bias_z = 0.0
        
        for tf in Timeframe:
            tf_key = tf.value
            candles = candles_by_tf.get(tf_key)
            if candles is None or len(candles) < 20:
                result.reasoning.append(f"⚠️ {tf_key}: Insufficient data.")
                continue
            
            last_candle = candles.iloc[-1]
            regime = market_physics.ingest_tick({
                'close': last_candle['close'],
                'tick_volume': last_candle['tick_volume'],
                'time': last_candle['time']
            })
            
            tf_score = analyze_hydrodynamics(tf, candles, regime, flow_bias_z)
            tf_scores[tf_key] = tf_score
            result.tf_details[tf_key] = tf_score
            
            if tf in [Timeframe.W1, Timeframe.D1, Timeframe.H4]:
                flow_bias_z += tf_score.ofi_z_score * 0.5
            
            directions.append(tf_score.direction)
            setattr(result, f"{tf_key.lower()}_score", tf_score.score)
        
        self._calculate_composite(result, tf_scores)
        
        if result.composite_score > 0.1:
            self._determine_trade(result, tf_scores, current_price)
            
        return result

    def _calculate_composite(self, result: MTFCompositeScore, tf_scores: Dict[str, TimeframeScore]):
        weighted_sum = 0.0
        total_weight = 0.0
        bullish = 0
        bearish = 0
        total = 0
        
        for tf in Timeframe:
            weight = get_tf_weight(tf)
            key = tf.value
            if key in tf_scores:
                s = tf_scores[key]
                weighted_sum += s.score * weight
                total_weight += weight
                if s.direction > 0: bullish += 1
                elif s.direction < 0: bearish += 1
                total += 1
        
        if total_weight > 0:
            result.composite_score = weighted_sum / total_weight
            
        if total > 0:
            result.alignment_percentage = (max(bullish, bearish) / total) * 100
            result.primary_direction = 1 if bullish > bearish else -1

        m1 = tf_scores.get("M1")
        h4 = tf_scores.get("H4")
        
        if m1 and m1.score >= settings.MTF_THRESHOLD_SNIPER_TRIGGER:
            h4_dir = h4.direction if h4 else 0
            if not ((h4_dir != 0) and (h4_dir != m1.direction)):
                boosted_score = max(result.composite_score, m1.score)
                if boosted_score >= settings.MTF_THRESHOLD_SNIPER:
                    result.composite_score = boosted_score
                    result.reasoning.append(f"🎯 Sniper Override: M1 Strong ({m1.score})")

    def _determine_trade(self, result: MTFCompositeScore, tf_scores: Dict[str, TimeframeScore], current_price: float):
        if result.composite_score < settings.MTF_THRESHOLD_SIGNAL: return
        result.action = "BUY" if result.primary_direction > 0 else "SELL"
        m1 = tf_scores.get("M1")
        m5 = tf_scores.get("M5")
        
        if (m1 and m1.is_vacuum) or (m5 and m5.is_vacuum):
            result.entry_type = "AGGRESSIVE_MARKET" 
            result.reasoning.append("Vacuum Run -> Aggressive Entry")
        elif (m1 and m1.is_absorption):
            result.entry_type = "PASSIVE_LIMIT"
            result.reasoning.append("Absorption -> Passive Entry")
        else:
            result.entry_type = "MARKET"
        
        atr_proxy = current_price * settings.MTF_ATR_PROXY_MULTIPLIER
        if result.action == "BUY":
            result.suggested_sl = current_price - (atr_proxy * settings.MTF_SL_ATR_MULTIPLIER)
            result.suggested_tp = current_price + (atr_proxy * settings.MTF_TP_ATR_MULTIPLIER)
        else:
            result.suggested_sl = current_price + (atr_proxy * settings.MTF_SL_ATR_MULTIPLIER)
            result.suggested_tp = current_price - (atr_proxy * settings.MTF_TP_ATR_MULTIPLIER)
        result.suggested_entry = current_price

mtf_engine = FractalMTFEngine()
