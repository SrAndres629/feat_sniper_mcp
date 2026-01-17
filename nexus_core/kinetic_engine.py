import numpy as np
import pandas as pd
from typing import Dict, List, Any

class KineticEngine:
    """
    [LEVEL 48] MULTIFRACTAL KINETIC ENGINE
    ======================================
    Implements the "Cloud Protocol" for visual simplification.
    Instead of 40 individual EMAs, we compute the STATE of 4 Layers:
    1. Micro (Red): Reactivity/Intent
    2. Structure (Green): Operational Path
    3. Macro (Blue): Memory/Equilibrium
    4. Bias (Gray): Absolute Regime
    """
    
    def __init__(self):
        # Layer Definitions (Loaded from Config for Singularity)
        from app.core.config import settings
        self.layers = {
            "micro": settings.LAYER_MICRO_PERIODS,
            "structure": settings.LAYER_OPERATIVE_PERIODS,
            "macro": settings.LAYER_MACRO_PERIODS
        }
        self.bias_period = settings.LAYER_BIAS_PERIOD
        # Thresholds
        self.knot_thresh = settings.KINETIC_COMPRESSION_THRESH
        self.expand_thresh = settings.KINETIC_EXPANSION_THRESH

    def compute_kinetic_state(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes the Multifractal State of the current market window.
        Returns a dictionary of normalized metrics for the Neural Net.
        """
        if df.empty:
            return {}

        close = df["close"]
        price = close.iloc[-1]
        
        # Calculate ATR for Normalization (Critical for Scale Invariance)
        high_low = df["high"] - df["low"]
        # ranges = pd.concat([high_low], axis=1) # Removed DF wrapping to avoid boolean ambiguity
        atr = high_low.rolling(14).mean().iloc[-1]
        
        if pd.isna(atr) or atr == 0: atr = price * 0.001
        atr = float(atr) # Ensure scalar

        metrics = {}
        centroids = {}
        slopes = {}

        # --- 1. CLOUD METRICS (Micro, Structure, Macro) ---
        for layer_name, periods in self.layers.items():
            ema_values = []
            
            # Compute all EMAs for this layer
            # [OPTIMIZATION] We need last 2 points for Slope Calculation
            for p in periods:
                series = close.ewm(span=p, adjust=False).mean()
                ema_values.append(series.iloc[-2:].values) # Get last 2
            
            ema_values = np.array(ema_values) # Shape (N_Periods, 2)
            
            # A. CENTROID (The "Center of Gravity" of the layer)
            # Mean across periods for T (current) and T-1 (prev)
            centroid_series = np.mean(ema_values, axis=0) # [Prev, Curr]
            centroid = centroid_series[1]
            prev_centroid = centroid_series[0]
            
            centroids[layer_name] = centroid
            
            # B. COMPRESSION (Girth/Fan)
            # Standard Deviation of the EMAs at current T
            spread = np.std(ema_values[:, 1])
            metrics[f"{layer_name}_compression"] = spread / atr # Normalized Girth
            
            # C. DISTANCE TO PRICE
            metrics[f"dist_{layer_name}"] = (price - centroid) / atr
            
            # D. SLOPE (Kinetic Intent)
            # Velocity of the Centroid
            slope = (centroid - prev_centroid) / atr
            metrics[f"{layer_name}_slope"] = slope
            slopes[layer_name] = slope

            # E. FASTEST MICRO TRACKING (Lag Reduction)
            if layer_name == "micro":
                # Assuming periods are sorted.
                # If Period 1 is present (Price), we want the next fastest (e.g. 2 or 3) to measure divergence.
                fast_idx = 0
                if len(periods) > 1 and periods[0] == 1:
                    fast_idx = 1
                
                fastest_ema = ema_values[fast_idx, 1]
                metrics["dist_micro_fast"] = (price - fastest_ema) / atr

        # --- 2. BIAS LINE (The 2048 SMMA) ---
        bias_series = close.ewm(alpha=1.0/self.bias_period, adjust=False).mean().iloc[-2:]
        bias_val = bias_series.iloc[-1]
        bias_prev = bias_series.iloc[-0]
        
        metrics["dist_bias"] = (price - bias_val) / atr
        metrics["bias_val"] = bias_val
        metrics["bias_slope"] = (bias_val - bias_prev) / atr
        
        # --- 3. INTER-LAYER RELATIONSHIPS (The "Change in Relation") ---
        
        # A. Micro vs Structure (Intent vs Reality)
        metrics["delta_micro_struct"] = (centroids["micro"] - centroids["structure"]) / atr
        
        # B. Structure vs Macro (Trend Strength)
        metrics["delta_struct_macro"] = (centroids["structure"] - centroids["macro"]) / atr
        
        # C. Alignment Score
        aligned_bull = (price > centroids["micro"] > centroids["structure"] > centroids["macro"])
        aligned_bear = (price < centroids["micro"] < centroids["structure"] < centroids["macro"])
        
        metrics["layer_alignment"] = 1.0 if aligned_bull else (-1.0 if aligned_bear else 0.0)
        
        # Export Centroids for Visualization if needed
        metrics["centroid_micro"] = centroids["micro"]
        metrics["centroid_struct"] = centroids["structure"]
        metrics["centroid_macro"] = centroids["macro"]
        metrics["bias_level"] = bias_val

        return metrics

    def detect_kinetic_patterns(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        [LEVEL 49] COGNITIVE PATTERN RECOGNITION (Refined V2)
        Decodes metrics into Semantic Patterns with Macro Awareness.
        """
        # 1. Unpack Metrics
        micro_comp = metrics.get("micro_compression", 1.0)
        struct_comp = metrics.get("structure_compression", 1.0)
        alignment = metrics.get("layer_alignment", 0.0)
        delta_m_s = metrics.get("delta_micro_struct", 0.0)
        dist_struct = metrics.get("dist_structure", 0.0)
        dist_macro = metrics.get("dist_macro", 0.0)
        
        # [LEVEL 50] Slopes for Regime Validation
        macro_slope = metrics.get("macro_slope", 0.0)
        struct_slope = metrics.get("structure_slope", 0.0)
        
        # 2. Define Thresholds 
        KNOT_THRESH = 0.3
        
        # 3. Pattern Logic
        pattern_id = 0 # 0: Unknown/Noise
        pattern_name = "NOISE"
        
        # PATTERN 1: EXPANSION (Clean Trend)
        if alignment != 0 and struct_comp > KNOT_THRESH:
            if (alignment > 0 and delta_m_s > 0) or (alignment < 0 and delta_m_s < 0):
                pattern_id = 1
                pattern_name = "EXPANSION_BULL" if alignment > 0 else "EXPANSION_BEAR"
        
        # PATTERN 2: COMPRESSION (Preparation)
        if micro_comp < KNOT_THRESH or struct_comp < KNOT_THRESH:
            if abs(alignment) < 1.0: 
                pattern_id = 2
                pattern_name = "COMPRESSION"
                
        # PATTERN 3: FALSE REVERSAL vs REGIME CHANGE
        # Scenario: Price Crossed Structure (e.g. Price < Struct but Macro is BULL)
        
        # Check if Price is fighting the Macro Trend
        fighting_macro_bull = (dist_macro > 0 and macro_slope > 0.05 and dist_struct < 0)
        fighting_macro_bear = (dist_macro < 0 and macro_slope < -0.05 and dist_struct > 0)
        
        if fighting_macro_bull or fighting_macro_bear:
            # Price broke structure, but Macro is still strong against it.
            # user_request: DEEP_RETRACEMENT instead of TRUE REVERSAL
            pattern_id = 3
            pattern_name = "DEEP_RETRACEMENT"
            
        # PATTERN 4: REGIME CHANGE (True)
        # Price broke Structure AND (Macro Broken OR Macro Flat/Turning)
        macro_broken_bull = (dist_macro < 0) # Price below Macro
        macro_broken_bear = (dist_macro > 0) # Price above Macro
        macro_flat = abs(macro_slope) < 0.02 # Macro lost steam
        
        # If we broke structure...
        if (dist_struct < 0 and struct_slope < 0) or (dist_struct > 0 and struct_slope > 0): 
            # ...check Macro validation
            valid_reversal_bear = (dist_struct < 0) and (macro_broken_bull or macro_flat)
            valid_reversal_bull = (dist_struct > 0) and (macro_broken_bear or macro_flat)
            
            if valid_reversal_bear:
                pattern_id = 4
                pattern_name = "REGIME_CHANGE_BEAR"
            elif valid_reversal_bull:
                pattern_id = 4
                pattern_name = "REGIME_CHANGE_BULL"

        # 4. Semantic State ("Who is in control?")
        d_m = abs(metrics.get("dist_micro_fast", metrics.get("dist_micro", 100))) # Use Fast
        d_s = abs(metrics.get("dist_structure", 100))
        d_M = abs(metrics.get("dist_macro", 100))
        
        control = "MICRO"
        if d_s < d_m and d_s < d_M: control = "STRUCTURE"
        if d_M < d_m and d_M < d_s: control = "MACRO"
        
        return {
            "pattern_id": pattern_id,
            "pattern_name": pattern_name,
            "control_layer": control,
            "kinetic_coherence": float(abs(alignment)),
            "layer_alignment": float(alignment) 
        }

# Singleton
kinetic_engine = KineticEngine()
