"""
Structure Engine - Probabilistic Market Structure Analysis
==========================================================
Implements:
- FourLayerEMA: 4-layer EMA system from CEMAs.mqh (Micro/Oper/Macro/Bias)
- StructuralConfidence: Probabilistic confidence scoring
- MAE Pattern Recognition: Momentum → Accumulation → Expansion
- BOS/CHOCH Detection: Break of Structure / Change of Character
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("feat.structure")


# =============================================================================
# PROBABILISTIC RESULT STRUCTURES
# =============================================================================

class EMALayer(Enum):
    """EMA Layer classification from CEMAs.mqh"""
    MICRO = "micro"           # Layer 1: Intent/Gas (fast EMAs)
    OPERATIVE = "operative"   # Layer 2: Structure/Water (medium EMAs)
    MACRO = "macro"           # Layer 3: Memory/Wall (slow EMAs)
    BIAS = "bias"             # Layer 4: Regime/Bedrock (SMMA 2048)


@dataclass
class LayerMetrics:
    """Metrics for a single EMA layer."""
    avg_value: float          # Average of all EMAs in layer
    spread: float             # Max - Min of layer
    compression: float        # Spread/Price * 1000 (squeeze indicator)
    slope: float              # Direction of layer movement
    layer_type: EMALayer


@dataclass
class StructuralConfidence:
    """Probabilistic result from structure analysis."""
    bos_confidence: float = 0.0       # 0.0-1.0: Break of Structure probability
    choch_confidence: float = 0.0     # 0.0-1.0: Change of Character probability
    zone_confidence: float = 0.0      # 0.0-1.0: Price at valid zone probability
    mae_confidence: float = 0.0       # 0.0-1.0: MAE pattern valid probability
    layer_alignment: float = 0.0      # 0.0-1.0: EMA layers aligned probability
    
    overall_form_score: float = 0.0   # Weighted combination
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bos_confidence": round(self.bos_confidence, 3),
            "choch_confidence": round(self.choch_confidence, 3),
            "zone_confidence": round(self.zone_confidence, 3),
            "mae_confidence": round(self.mae_confidence, 3),
            "layer_alignment": round(self.layer_alignment, 3),
            "overall_form_score": round(self.overall_form_score, 3),
            "reasoning": self.reasoning
        }


# =============================================================================
# FOUR-LAYER EMA SYSTEM (from CEMAs.mqh)
# =============================================================================

class FourLayerEMA:
    """
    4-Layer EMA System - Institutional Grade Market Physics.
    
    Ported from CEMAs.mqh:
    - Layer 1 (Micro/Gas): EMAs [1,2,3,6,7,8,9,12,13,14] → Intent
    - Layer 2 (Operative/Water): EMAs [16,24,32,48,64,96,128,160,192,224] → Structure
    - Layer 3 (Macro/Wall): EMAs [256,320,384,448,512,640,768,896,1024,1280] → Memory
    - Layer 4 (Bias/Bedrock): SMMA 2048 → Regime
    
    Returns probabilistic confidence based on layer alignment and position.
    """
    
    # Layer periods from CEMAs.mqh
    MICRO_PERIODS = [1, 2, 3, 6, 7, 8, 9, 12, 13, 14]
    OPERATIVE_PERIODS = [16, 24, 32, 48, 64, 96, 128, 160, 192, 224]
    MACRO_PERIODS = [256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280]
    BIAS_PERIOD = 2048
    
    def __init__(self):
        logger.info("[FourLayerEMA] Multifractal Layer Physics initialized")
    
    def compute_layer_metrics(
        self, 
        df: pd.DataFrame, 
        layer: EMALayer
    ) -> Optional[LayerMetrics]:
        """
        Compute metrics for a specific EMA layer.
        
        Returns:
            LayerMetrics with avg, spread, compression, slope
        """
        if "close" not in df.columns or len(df) < 20:
            return None
        
        close = df["close"]
        
        # Select periods based on layer
        if layer == EMALayer.MICRO:
            periods = self.MICRO_PERIODS
        elif layer == EMALayer.OPERATIVE:
            periods = self.OPERATIVE_PERIODS
        elif layer == EMALayer.MACRO:
            periods = [p for p in self.MACRO_PERIODS if p <= len(df)]
        elif layer == EMALayer.BIAS:
            periods = [min(self.BIAS_PERIOD, len(df))]
        else:
            return None
        
        if not periods:
            return None
        
        # Calculate EMAs for each period in layer
        ema_values = []
        for p in periods:
            if p <= len(df):
                ema = close.ewm(span=p, adjust=False).mean().iloc[-1]
                ema_values.append(ema)
        
        if not ema_values:
            return None
        
        # Calculate metrics
        avg_value = np.mean(ema_values)
        spread = max(ema_values) - min(ema_values) if len(ema_values) > 1 else 0
        compression = (spread / avg_value * 1000) if avg_value > 0 else 0
        
        # Stable Slope: Use Linear Regression over last 3-5 bars to filter M1 noise
        window = 5
        if len(df) >= window:
            # Use numpy for stable linear regression slope
            y = []
            for i in range(1, window + 1):
                # Calculate avg for each of the last 'window' bars
                past_emas = [close.ewm(span=p, adjust=False).mean().iloc[-i] 
                             for p in periods if p <= len(df)]
                y.insert(0, np.mean(past_emas) if past_emas else avg_value)
            
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
        else:
            slope = 0.0
        
        return LayerMetrics(
            avg_value=avg_value,
            spread=spread,
            compression=compression,
            slope=slope,
            layer_type=layer
        )
    
    def get_all_layer_metrics(self, df: pd.DataFrame) -> Dict[EMALayer, LayerMetrics]:
        """Compute metrics for all 4 layers."""
        result = {}
        for layer in EMALayer:
            metrics = self.compute_layer_metrics(df, layer)
            if metrics:
                result[layer] = metrics
        return result
    
    def get_price_position(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine where price sits relative to all layers.
        
        Returns:
            Dict with position info and confidence score
        """
        if len(df) < 20:
            return {"position": "WARMUP", "confidence": 0.0}
        
        price = df["close"].iloc[-1]
        metrics = self.get_all_layer_metrics(df)
        
        # Determine position relative to operative layer (key structure)
        oper = metrics.get(EMALayer.OPERATIVE)
        if not oper:
            return {"position": "UNKNOWN", "confidence": 0.0}
        
        # Position relative to operative layer
        if price > oper.avg_value + oper.spread:
            position = "ABOVE_STRUCTURE"
            base_conf = 0.7
        elif price < oper.avg_value - oper.spread:
            position = "BELOW_STRUCTURE"
            base_conf = 0.7
        else:
            position = "INSIDE_STRUCTURE"
            base_conf = 0.5
        
        # Confidence boost if layers are aligned
        alignment_conf = self.compute_layer_alignment(df)
        
        return {
            "position": position,
            "price": price,
            "operative_avg": oper.avg_value,
            "confidence": min(1.0, base_conf + alignment_conf * 0.3)
        }
    
    def compute_layer_alignment(self, df: pd.DataFrame) -> float:
        """
        Compute alignment score between layers (0.0-1.0).
        
        Perfect alignment (1.0): All layers sloping same direction, 
        Micro > Oper > Macro (bullish) or Micro < Oper < Macro (bearish).
        """
        metrics = self.get_all_layer_metrics(df)
        
        if len(metrics) < 3:
            return 0.0
        
        micro = metrics.get(EMALayer.MICRO)
        oper = metrics.get(EMALayer.OPERATIVE)
        macro = metrics.get(EMALayer.MACRO)
        
        if not all([micro, oper, macro]):
            return 0.0
        
        # Check slope alignment
        slopes = [micro.slope, oper.slope, macro.slope]
        all_positive = all(s > 0 for s in slopes)
        all_negative = all(s < 0 for s in slopes)
        slope_aligned = 1.0 if (all_positive or all_negative) else 0.3
        
        # Check layer ordering (bullish: micro > oper > macro)
        bullish_order = micro.avg_value > oper.avg_value > macro.avg_value
        bearish_order = micro.avg_value < oper.avg_value < macro.avg_value
        order_aligned = 1.0 if (bullish_order or bearish_order) else 0.4
        
        # Combine
        return (slope_aligned * 0.5 + order_aligned * 0.5)
    
    def is_cloud_compressed(self, df: pd.DataFrame, threshold: float = 0.5) -> bool:
        """
        Check if operative layer is compressed (squeeze indicator).
        
        Compression < threshold indicates potential breakout.
        """
        oper = self.compute_layer_metrics(df, EMALayer.OPERATIVE)
        return oper is not None and oper.compression < threshold


# Global singleton
four_layer_ema = FourLayerEMA()


class MAE_Pattern_Recognizer:
    """
    Gate F: Pattern Recognition Cortex.
    Implements the MAE Axiom: Momentum -> Accumulation -> Expansion.
    Detects structural shifts (BOS/CHOCH) and fractal pivots.
    """

    def __init__(self):
        logger.info("[Form] Pattern Engine Online (MAE Analysis Active)")

    def detect_mae_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detects Momentum -> Accumulation -> Expansion (MAE) phases.
        """
        if len(df) < 15:
            return {"phase": "WARMUP", "status": "RANGING"}

        # ATR Proxy for normalization
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]

        # 1. Momentum: Large body displacement
        body = df["close"].iloc[-1] - df["open"].iloc[-1]
        is_momentum = abs(body) > (atr * 1.5)

        # 2. Accumulation: Compressed range (last 3-5 candles)
        recent_range = df["high"].iloc[-5:-1].max() - df["low"].iloc[-5:-1].min()
        is_accumulation = recent_range < (atr * 1.2)

        # 3. Expansion: Breaking the accumulation zone
        upper_bound = df["high"].iloc[-5:-1].max()
        lower_bound = df["low"].iloc[-5:-1].min()

        is_expansion_up = df["close"].iloc[-1] > upper_bound and body > 0
        is_expansion_down = df["close"].iloc[-1] < lower_bound and body < 0

        status = "RANGING"
        phase = "NORMAL"

        if is_momentum:
            phase = "MOMENTUM"
            status = "IMPULSE"
        elif is_accumulation:
            phase = "ACCUMULATION"
            status = "COMPRESSION"
        elif is_expansion_up or is_expansion_down:
            phase = "EXPANSION"
            status = "BREAKOUT"

        return {
            "phase": phase,
            "status": status,
            "is_expansion": is_expansion_up or is_expansion_down,
            "direction": 1 if is_expansion_up else (-1 if is_expansion_down else 0),
        }


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

    def identify_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bill Williams Fractals (5-candle pattern).
        A fractal is confirmed when the middle candle is the highest/lowest of 5.
        """
        # High Fractals
        df["fractal_high"] = (
            (df["high"].shift(2) > df["high"].shift(4))
            & (df["high"].shift(2) > df["high"].shift(3))
            & (df["high"].shift(2) > df["high"].shift(1))
            & (df["high"].shift(2) > df["high"])
        )

        # Low Fractals
        df["fractal_low"] = (
            (df["low"].shift(2) < df["low"].shift(4))
            & (df["low"].shift(2) < df["low"].shift(3))
            & (df["low"].shift(2) < df["low"].shift(1))
            & (df["low"].shift(2) < df["low"])
        )
        return df

    def detect_structural_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BOS (Break of Structure): Trend continuation.
        CHOCH (Change of Character): Trend reversal (First break against trend).
        """
        df = self.identify_fractals(df)

        # Track last confirmed fractal levels
        df["last_h_fractal"] = df["high"].where(df["fractal_high"]).ffill()
        df["last_l_fractal"] = df["low"].where(df["fractal_low"]).ffill()

        # BOS: Close exceeds last fractal in trend direction
        df["bos_bull"] = (df["close"] > df["last_h_fractal"].shift(1)) & (
            df["close"].shift(1) <= df["last_h_fractal"].shift(1)
        )
        df["bos_bear"] = (df["close"] < df["last_l_fractal"].shift(1)) & (
            df["close"].shift(1) >= df["last_l_fractal"].shift(1)
        )

        # CHOCH (Change of Character): Reversal - 1st break against current trend
        # We define trend based on Operative (L2) vs Macro (L3)
        # For M1 Sniper, we prioritize speed: Pure fractal break
        df["choch_bull"] = df["bos_bull"] & (df["close"].shift(1) < df["last_l_fractal"].shift(1))
        df["choch_bear"] = df["bos_bear"] & (df["close"].shift(1) > df["last_h_fractal"].shift(1))
        
        return df

    def compute_feat_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final FEAT Index calculation (0-100).
        """
        df = self.detect_structural_shifts(df)

        # Scoring logic
        mae = self.mae_recognizer.detect_mae_pattern(df)

        # F-Score (Form)
        f_score = 0.0
        if mae["phase"] == "EXPANSION":
            f_score += 0.4
        if df["bos_bull"].iloc[-1] or df["bos_bear"].iloc[-1]:
            f_score += 0.3
        if df["choch_bull"].iloc[-1] or df["choch_bear"].iloc[-1]:
            f_score += 0.3

        # E-Score (Space) - Proxy using FVG from OHLC
        fvg_bull = (df["low"] > df["high"].shift(2)).iloc[-1]
        fvg_bear = (df["high"] < df["low"].shift(2)).iloc[-1]
        e_score = 0.5 if (fvg_bull or fvg_bear) else 0.2

        # A-Score (Acceleration) - Simple volatility proxy
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        curr_range = df["high"].iloc[-1] - df["low"].iloc[-1]
        a_score = min(1.0, curr_range / (atr * 2 + 1e-9))

        # T-Score (Time) - Handled externally, but adding dummy for index
        t_score = 0.5

        w = self.config["weights"]
        feat_val = (
            w["F"] * f_score + w["E"] * e_score + w["A"] * a_score + w["T"] * t_score
        )

        res = pd.DataFrame(index=df.index)
        res["feat_index"] = round(feat_val * 100, 2)
        res["structure_status"] = mae["status"]
        res["is_mae_expansion"] = mae["is_expansion"]

        return res

    def get_structural_narrative(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns the latest structural narrative: BOS/CHOCH levels and types.
        """
        df = self.detect_structural_shifts(df)
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
            "status": self.mae_recognizer.detect_mae_pattern(df)["status"],
        }

    def detect_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Supply/Demand zones based on fractal clustering with Time-Based Decay.
        
        A zone is formed when 2+ fractals occur within ATR distance.
        Zone strength = (touches / 5.0) * decay_factor * [PVP_BOOST]
        """
        if "close" not in df.columns or len(df) < 20:
            df["zone_type"] = "NONE"
            df["zone_high"] = 0.0
            df["zone_low"] = 0.0
            df["zone_strength"] = 0.0
            return df
        
        # [PVP INTEGRATION]
        from app.skills.volume_profile import volume_profile
        profile = volume_profile.get_profile(df, bins=50)

        # Ensure fractals are identified
        df = self.identify_fractals(df)
        
        # ATR for zone width tolerance
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        zone_tolerance = atr * 0.5 if atr > 0 else 0.0001
        
        # Initial state
        df["zone_type"] = "NONE"
        df["zone_high"] = 0.0
        df["zone_low"] = 0.0
        df["zone_strength"] = 0.0
        
        current_idx = len(df) - 1
        
        # History lookback for zone detection (last 100 bars)
        lookback = min(100, len(df))
        history = df.tail(lookback)
        
        # 1. Supply Zones (High Fractal Clusters)
        high_indices = history.index[history["fractal_high"]].tolist()
        if len(high_indices) >= 2:
            # Check most recent high fractals first
            for idx in reversed(high_indices):
                level = df.at[idx, "high"]
                # Find other high fractals near this level
                matches = [i for i in high_indices if abs(df.at[i, "high"] - level) < zone_tolerance]
                touches = len(matches)
                if touches >= 2:
                    # Supply zone confirmed
                    # Get integer position for age calculation
                    idx_pos = df.index.get_loc(idx)
                    age = current_idx - idx_pos
                    decay = 1.0 / (1.0 + 0.01 * age) # 1% force loss per bar
                    
                    # [PHYSICS] Volume Density Boost
                    vol_boost = volume_profile.get_zone_quality(level+zone_tolerance, level-zone_tolerance, profile)
                    
                    df.loc[df.index[-1], "zone_type"] = "SUPPLY"
                    df.loc[df.index[-1], "zone_high"] = level + zone_tolerance
                    df.loc[df.index[-1], "zone_low"] = level - zone_tolerance
                    
                    # Final Strength Calculation
                    raw_strength = min(1.0, (touches / 5.0))
                    df.loc[df.index[-1], "zone_strength"] = min(1.0, raw_strength * decay * vol_boost)
                    break
                    
        # 2. Demand Zones (Low Fractal Clusters) - only check if no supply zone detected
        if df.iloc[-1]["zone_type"] == "NONE":
            low_indices = history.index[history["fractal_low"]].tolist()

            if len(low_indices) >= 2:
                for idx in reversed(low_indices):
                    level = df.at[idx, "low"]
                    matches = [i for i in low_indices if abs(df.at[i, "low"] - level) < zone_tolerance]
                    touches = len(matches)
                    if touches >= 2:
                        # Get integer position for age calculation
                        idx_pos = df.index.get_loc(idx)
                        age = current_idx - idx_pos
                        decay = 1.0 / (1.0 + 0.01 * age)
                        
                        # [PHYSICS] Volume Density Boost
                        vol_boost = volume_profile.get_zone_quality(level+zone_tolerance, level-zone_tolerance, profile)
                        
                        df.loc[df.index[-1], "zone_type"] = "DEMAND"
                        df.loc[df.index[-1], "zone_high"] = level + zone_tolerance
                        df.loc[df.index[-1], "zone_low"] = level - zone_tolerance
                        
                        # Final Strength Calculation
                        raw_strength = min(1.0, (touches / 5.0))
                        df.loc[df.index[-1], "zone_strength"] = min(1.0, raw_strength * decay * vol_boost)
                        break
        
        return df

    def get_zone_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns the status of the nearest Supply/Demand zone.
        """
        df = self.detect_zones(df)
        last_row = df.iloc[-1]
        
        if last_row["zone_type"] == "NONE":
            return {
                "nearest_zone": "NONE",
                "distance_to_zone": 0.0,
                "zone_strength": 0.0,
                "is_in_zone": False
            }
        
        # Calculate distance from current price to zone
        current_price = last_row["close"]
        zone_high = last_row["zone_high"]
        zone_low = last_row["zone_low"]
        
        if last_row["zone_type"] == "SUPPLY":
            distance = zone_high - current_price
            is_in_zone = current_price <= zone_high and current_price >= zone_low
        else:  # DEMAND
            distance = current_price - zone_low
            is_in_zone = current_price <= zone_high and current_price >= zone_low
        
        return {
            "nearest_zone": last_row["zone_type"],
            "distance_to_zone": float(distance),
            "zone_strength": float(last_row["zone_strength"]),
            "is_in_zone": is_in_zone
        }

    def get_structural_health(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns a health score for the current structure.
        
        Health = (BOS Strength + Zone Strength + Trend Alignment) / 3
        
        Returns:
            dict with health score and contributing factors
        """
        df = self.detect_structural_shifts(df)
        df = self.detect_zones(df)
        
        last_row = df.iloc[-1]
        
        # 1. BOS Strength (0-1)
        bos_strength = 0.0
        if last_row["bos_bull"] or last_row["bos_bear"]:
            bos_strength = 0.8
        elif last_row["choch_bull"] or last_row["choch_bear"]:
            bos_strength = 0.5
        
        # 2. Zone Strength (0-1)
        zone_strength = last_row["zone_strength"]
        
        # 3. Trend Alignment (0-1)
        # Check if price is above/below recent fractals
        trend_alignment = 0.5
        if last_row["fractal_high"] and last_row["fractal_low"]:
            if last_row["close"] > last_row["last_h_fractal"]:
                trend_alignment = 0.8  # Uptrend
            elif last_row["close"] < last_row["last_l_fractal"]:
                trend_alignment = 0.8  # Downtrend
            else:
                trend_alignment = 0.5  # Neutral
        
        # Calculate overall health
        health_score = (bos_strength + zone_strength + trend_alignment) / 3
        
        return {
            "health_score": round(health_score, 2),
            "bos_strength": round(bos_strength, 2),
            "zone_strength": round(zone_strength, 2),
            "trend_alignment": round(trend_alignment, 2),
            "status": "HEALTHY" if health_score > 0.6 else 
                      "NEUTRAL" if health_score > 0.3 else "WEAK"
        }

    def get_structural_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns a comprehensive structural report.
        """
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

    def get_structural_summary(self, df: pd.DataFrame) -> str:
        """
        Returns a human-readable summary of the current structure.
        """
        report = self.get_structural_report(df)
        
        # Build summary string
        summary = f"Structural Report:\n"
        summary += f"- MAE Phase: {report['mae_pattern']['phase']} ({report['mae_pattern']['status']})\n"
        summary += f"- Health Score: {report['health']['health_score']} ({report['health']['status']})\n"
        summary += f"- Nearest Zone: {report['zones']['nearest_zone']} (Strength: {report['zones']['zone_strength']})\n"
        summary += f"- Trend Alignment: {report['health']['trend_alignment']}\n"
        
        return summary

    def get_structural_score(self, df: pd.DataFrame) -> float:
        """
        Returns a single structural score (0-100).
        """
        report = self.get_structural_report(df)
        
        # Weighted score calculation
        score = (
            report['health']['health_score'] * 0.4 +
            report['zones']['zone_strength'] * 0.3 +
            report['mae_pattern']['is_expansion'] * 0.2 +
            report['health']['trend_alignment'] * 0.1
        ) * 100
        
        return round(float(score), 2)

    def get_structural_risk(self, df: pd.DataFrame) -> float:
        """
        Returns a risk score (0-100).
        
        Risk = (MAE Volatility + Zone Proximity + Trend Strength) / 3
        
        Returns:
            dict with risk score and contributing factors
        """
        report = self.get_structural_report(df)
        
        # 1. MAE Volatility (0-1)
        # Higher volatility = higher risk
        mae_volatility = 0.0
        if report['mae_pattern']['phase'] == "EXPANSION":
            mae_volatility = 0.8
        elif report['mae_pattern']['phase'] == "ACCUMULATION":
            mae_volatility = 0.5
        elif report['mae_pattern']['phase'] == "DISTRIBUTION":
            mae_volatility = 0.5
        elif report['mae_pattern']['phase'] == "CONTRACTION":
            mae_volatility = 0.2
        
        # 2. Zone Proximity (0-1)
        # Closer to zone = higher risk
        zone_proximity = 1.0 - report['zones']['zone_strength']
        
        # 3. Trend Strength (0-1)
        # Weaker trend = higher risk
        trend_risk = 1.0 - report['health']['trend_alignment']
        
        # Calculate overall risk
        risk_score = (mae_volatility + zone_proximity + trend_risk) / 3
        
        return round(float(risk_score * 100), 2)

    def get_structural_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Returns a list of potential trading opportunities.
        
        Returns:
            list of opportunity dicts with:
                - type: "BUY" or "SELL"
                - zone: "SUPPLY" or "DEMAND"
                - confidence: 0-1
                - risk_reward_ratio: float
                - reason: str
        """
        report = self.get_structural_report(df)
        opportunities = []
        
        # Check for bullish opportunities
        if report['mae_pattern']['phase'] == "ACCUMULATION" and report['zones']['nearest_zone'] == "DEMAND":
            opportunities.append({
                "type": "BUY",
                "zone": "DEMAND",
                "confidence": round(report['health']['health_score'] * 0.8, 2),
                "risk_reward_ratio": 1.0 / (1.0 - report['zones']['zone_strength']),
                "reason": "Price in Demand zone after Accumulation"
            })
        
        # Check for bearish opportunities
        if report['mae_pattern']['phase'] == "DISTRIBUTION" and report['zones']['nearest_zone'] == "SUPPLY":
            opportunities.append({
                "type": "SELL",
                "zone": "SUPPLY",
                "confidence": round(report['health']['health_score'] * 0.8, 2),
                "risk_reward_ratio": 1.0 / (1.0 - report['zones']['zone_strength']),
                "reason": "Price in Supply zone after Distribution"
            })
        
        # Check for trend continuation opportunities
        if report['health']['trend_alignment'] > 0.7:
            if report['zones']['nearest_zone'] == "DEMAND":
                opportunities.append({
                    "type": "BUY",
                    "zone": "DEMAND",
                    "confidence": round(report['health']['health_score'] * 0.6, 2),
                    "risk_reward_ratio": 1.0 / (1.0 - report['zones']['zone_strength']),
                    "reason": "Trend continuation in Demand zone"
                })
            elif report['zones']['nearest_zone'] == "SUPPLY":
                opportunities.append({
                    "type": "SELL",
                    "zone": "SUPPLY",
                    "confidence": round(report['health']['health_score'] * 0.6, 2),
                    "risk_reward_ratio": 1.0 / (1.0 - report['zones']['zone_strength']),
                    "reason": "Trend continuation in Supply zone"
                })
        
        return opportunities

    def get_structural_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns a complete trading strategy based on current structure.
        """
        report = self.get_structural_report(df)
        opportunities = self.get_structural_opportunities(df)
        
        # Select best opportunity
        best_opportunity = None
        if opportunities:
            best_opportunity = max(opportunities, key=lambda x: x['confidence'])
        
        # Determine strategy type
        strategy_type = "NEUTRAL"
        if best_opportunity:
            if best_opportunity['confidence'] > 0.7:
                strategy_type = "STRONG_OPPORTUNITY"
            elif best_opportunity['confidence'] > 0.5:
                strategy_type = "MODERATE_OPPORTUNITY"
            else:
                strategy_type = "WEAK_OPPORTUNITY"
        
        return {
            "strategy_type": strategy_type,
            "best_opportunity": best_opportunity,
            "all_opportunities": opportunities,
            "structural_health": report['health'],
            "mae_pattern": report['mae_pattern'],
            "zones": report['zones']
        }

    def get_structural_risk_reward(self, df: pd.DataFrame) -> float:
        """
        Returns the risk/reward ratio for the current structure.
        
        Returns:
            float: risk/reward ratio (higher is better)
        """
        report = self.get_structural_report(df)
        
        # Calculate risk/reward based on zone strength
        if report['zones']['zone_strength'] > 0:
            risk_reward = 1.0 / (1.0 - report['zones']['zone_strength'])
        else:
            risk_reward = 0.0
        
        return round(float(risk_reward), 2)

    def get_structural_bias(self, df: pd.DataFrame) -> str:
        """
        Returns the overall market bias based on structure.
        
        Returns:
            str: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        report = self.get_structural_report(df)
        
        # Calculate bias based on trend alignment and zone strength
        if report['health']['trend_alignment'] > 0.6 and report['zones']['zone_strength'] > 0.5:
            return "BULLISH"
        elif report['health']['trend_alignment'] < 0.4 and report['zones']['zone_strength'] > 0.5:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def get_structural_zones(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Returns all detected supply and demand zones.
        
        Returns:
            list of zone dicts with:
                - type: "SUPPLY" or "DEMAND"
                - level: price level
                - strength: 0-1
                - distance_from_price: float
        """
        report = self.get_structural_report(df)
        return report['zones']['zones']

    def get_structural_trend(self, df: pd.DataFrame) -> str:
        """
        Returns the current trend direction.
        
        Returns:
            str: "BULLISH", "BEARISH", or "NEUTRAL"
        """
        report = self.get_structural_report(df)
        return report['health']['trend']

    def get_structural_trend_alignment(self, df: pd.DataFrame) -> float:
        """
        Returns the trend alignment score (0-1).
        
        Returns:
            float: trend alignment score
        """
        report = self.get_structural_report(df)
        return report['health']['trend_alignment']


structure_engine = StructureEngine()
