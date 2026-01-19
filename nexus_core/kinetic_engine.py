import numpy as np
import pandas as pd
from typing import Dict, List, Any

class KineticValidator:
    """
    [DOCTORAL SKILL: ABSORPTION TEST]
    Validates if an Institutional Impulse is real or a fakeout.
    """
    
    def calculate_feat_force(self, candle: pd.Series, atr: float, rvol: float) -> float:
        """
        Fórmula Obligatoria: Force = (Body_Size / (ATR + 1e-9)) * Relative_Volume
        """
        body_size = abs(candle["close"] - candle["open"])
        force = (body_size / (atr + 1e-9)) * rvol
        return force

    def check_absorption_state(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        State Machine for Absorption Logic.
        Returns:
            - state: "NEUTRAL", "IMPULSE", "MONITORING", "CONFIRMED", "FAILED"
            - progress: (int) candles checked (0-3)
        """
        if len(df) < 5: 
            return {"state": "NEUTRAL", "progress": 0, "feat_force": 0.0}
        
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        if atr == 0: atr = 1e-9
        
        # We need to simulate the state machine by looking back.
        # Find the most recent "Active" Impulse (Force > 2.0) within last 4 candles.
        
        active_impulse_idx = None
        
        # Check T (current) back to T-3
        for i in range(0, 4): 
            idx = -(i + 1) # -1, -2, -3, -4
            candle = df.iloc[idx]
            
            # Calculate RVOL (Simple Proxy) for historical inspection
            # Ideally passed in, but we'll calculate local
            vol_mean = df["volume"].iloc[idx-20:idx].mean() if abs(idx) > 20 else df["volume"].mean()
            rvol = candle["volume"] / (vol_mean + 1e-9)
            
            force = self.calculate_feat_force(candle, atr, rvol)
            
            if force > 2.0:
                # Found an impulse
                active_impulse_idx = idx
                break
        
        if active_impulse_idx is None:
             # Check current candle specifically if it's an impulse
             curr = df.iloc[-1]
             # rvol for current
             vol_mean = df["volume"].iloc[-21:-1].mean()
             rvol = curr["volume"] / (vol_mean + 1e-9)
             force = self.calculate_feat_force(curr, atr, rvol)
             if force > 2.0:
                 return {"state": "IMPULSE", "progress": 0, "feat_force": force}
             return {"state": "NEUTRAL", "progress": 0, "feat_force": force}

        # We have an impulse at active_impulse_idx
        impulse_candle = df.iloc[active_impulse_idx]
        impulse_body = abs(impulse_candle["close"] - impulse_candle["open"])
        is_bull = impulse_candle["close"] > impulse_candle["open"]
        limit_level = impulse_candle["low"] + (impulse_body * 0.5) if is_bull else impulse_candle["high"] - (impulse_body * 0.5)
        
        # Candles since impulse
        # idx is like -3. So candles are -2, -1. (2 candles passed)
        candles_passed = abs(active_impulse_idx) - 1
        
        # Validate strict 50% defense
        subsequent = df.iloc[active_impulse_idx+1:]
        
        for _, fut in subsequent.iterrows():
            if is_bull:
                if fut["close"] < limit_level:
                    return {"state": "FAILED", "progress": candles_passed, "feat_force": 0.0}
            else:
                if fut["close"] > limit_level:
                    return {"state": "FAILED", "progress": candles_passed, "feat_force": 0.0}
                    
        if candles_passed >= 3:
            return {"state": "CONFIRMED", "progress": 3, "feat_force": 0.0}
            
        return {"state": "MONITORING", "progress": candles_passed, "feat_force": 0.0}

class SpectralMechanics:
    """
    [LEVEL 51] SPECTRAL TENSOR ENCODING
    Translates visual 'Color Gradients' into Neural Tensors.
    """
    def __init__(self):
        # 🟥 GRUPO 1: MICRO-INTENCIÓN
        self.sub_1 = [1, 2, 3]
        self.sub_2 = [6, 7, 8, 9]
        self.sub_3 = [12, 13, 14]
        
        # 🟨 GRUPO 2: OPERATIVA / AGUA
        self.sub_4 = [16, 24, 32]
        self.sub_5 = [48, 64, 96]
        self.sub_6 = [128, 160, 192, 224]
        
        # 🟩 GRUPO 3: MACRO / MURO
        self.sub_7 = [256, 320, 384]
        self.sub_8 = [448, 512, 640]
        self.sub_9 = [768, 896, 1024, 1280]
        
        # ⬛ GRUPO 4: SESGO / ROCA
        self.sub_10 = [2048]

    def compute_group_integrity(self, emas: List[float]) -> float:
        """
        Calculates Gradient Integrity.
        +1.0: Perfect Bullish (Fast > Slow)
        -1.0: Perfect Bearish (Fast < Slow)
        0.0: Disordered (Choppy)
        """
        if len(emas) < 2: return 0.0
        
        bull_pairs = 0
        bear_pairs = 0
        total_pairs = len(emas) - 1
        
        for i in range(total_pairs):
            if emas[i] > emas[i+1]:
                bull_pairs += 1
            elif emas[i] < emas[i+1]:
                bear_pairs += 1
                
        # Normalize
        if bull_pairs == total_pairs: return 1.0
        if bear_pairs == total_pairs: return -1.0
        
        # Partial Score
        return (bull_pairs - bear_pairs) / total_pairs

    def compute_chromatic_divergence(self, emas: List[float], atr: float) -> float:
        """
        Calculates the width of the spectral band (Chromatic Divergence).
        """
        if not emas: return 0.0
        width = max(emas) - min(emas)
        return width / (atr + 1e-9)

    def analyze_spectrum(self, close_series: pd.Series, atr: float = 1.0) -> Dict[str, float]:
        """
        Computes the Integrity Tensor for all 10 layers.
        """
        metrics = {}
        
        # Helper to get EMA values (last point)
        def get_emas(periods):
            vals = []
            for p in periods:
                vals.append(close_series.ewm(span=p, adjust=False).mean().iloc[-1])
            return vals

        # 1. MICRO INTEGRITY (Sub 1, 2, 3)
        s1 = get_emas(self.sub_1)
        s2 = get_emas(self.sub_2)
        s3 = get_emas(self.sub_3)
        
        int_s1 = self.compute_group_integrity(s1)
        int_s2 = self.compute_group_integrity(s2)
        int_s3 = self.compute_group_integrity(s3)
        
        metrics["integrity_sub1"] = int_s1
        metrics["integrity_sub2"] = int_s2
        metrics["integrity_sub3"] = int_s3
        
        # Global Micro Integrity
        metrics["integrity_micro"] = (int_s1 * 0.5) + (int_s2 * 0.3) + (int_s3 * 0.2)
        div_m = self.compute_chromatic_divergence(s1 + s2 + s3, atr)
        metrics["micro_spectrum"] = metrics["integrity_micro"] * div_m
        
        # 2. OPERATIVE INTEGRITY (Sub 4, 5, 6)
        s4 = get_emas(self.sub_4)
        s5 = get_emas(self.sub_5)
        s6 = get_emas(self.sub_6)
        
        int_s4 = self.compute_group_integrity(s4)
        int_s5 = self.compute_group_integrity(s5)
        int_s6 = self.compute_group_integrity(s6)
        
        metrics["integrity_sub4"] = int_s4
        metrics["integrity_sub5"] = int_s5
        metrics["integrity_sub6"] = int_s6
        
        metrics["integrity_structure"] = (int_s4 * 0.4 + int_s5 * 0.4 + int_s6 * 0.2)
        div_o = self.compute_chromatic_divergence(s4 + s5 + s6, atr)
        metrics["operative_spectrum"] = metrics["integrity_structure"] * div_o
                                          
        # 3. MACRO INTEGRITY (Sub 7, 8, 9)
        s7 = get_emas(self.sub_7)
        s8 = get_emas(self.sub_8)
        s9 = get_emas(self.sub_9)
        
        int_s7 = self.compute_group_integrity(s7)
        int_s8 = self.compute_group_integrity(s8)
        int_s9 = self.compute_group_integrity(s9)
        
        metrics["integrity_sub7"] = int_s7
        metrics["integrity_sub8"] = int_s8
        metrics["integrity_sub9"] = int_s9

        metrics["integrity_macro"] = (int_s7 + int_s8 + int_s9) / 3.0
        div_M = self.compute_chromatic_divergence(s7 + s8 + s9, atr)
        metrics["macro_spectrum"] = metrics["integrity_macro"] * div_M

        # 4. BIAS (Sub 10)
        s10 = get_emas(self.sub_10)
        metrics["bias_level"] = s10[0]
                                      
        return metrics

class KineticEngine:
    """
    [LEVEL 48] MULTIFRACTAL KINETIC ENGINE
    ...
    """
    
    def __init__(self):
        # Layer Definitions
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
        
        # [DOCTORAL]
        self.validator = KineticValidator()
        self.spectral = SpectralMechanics()

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
            # [OPTIMIZATION] We need last 3 points for Acceleration (2nd Derivative)
            for p in periods:
                series = close.ewm(span=p, adjust=False).mean()
                ema_values.append(series.iloc[-3:].values) # Get last 3: [T-2, T-1, T]
            
            ema_values = np.array(ema_values) # Shape (N_Periods, 3)
            
            # A. CENTROIDS
            # Mean across periods for T, T-1, T-2
            centroid_series = np.mean(ema_values, axis=0) # [T-2, T-1, T]
            centroid = centroid_series[2]
            prev_centroid = centroid_series[1]
            prev2_centroid = centroid_series[0]
            
            centroids[layer_name] = centroid
            
            # B. COMPRESSION (Girth/Fan)
            # Standard Deviation of the EMAs at current T
            spread = np.std(ema_values[:, 2])
            metrics[f"{layer_name}_compression"] = spread / atr # Normalized Girth
            
            # C. DISTANCE TO PRICE
            metrics[f"dist_{layer_name}"] = (price - centroid) / atr
            
            # ema_values has [T-1, T]. So we only have 2 points.
            # To calc acceleration we need 3 points: T, T-1, T-2.
            # I will assume `prev_slope` is stored or approximating with limited history?
            # Actually, `ema_values` comes from `close.ewm...iloc[-2:]`. 
            # I should fetch 3 points to calc Accel.
            
            # F. FASTEST MICRO TRACKING (Lag Reduction)
            if layer_name == "micro":
                # Assuming periods are sorted.
                # If Period 1 is present (Price), we want the next fastest (e.g. 2 or 3) to measure divergence.
                fast_idx = 0
                if len(periods) > 1 and periods[0] == 1:
                    fast_idx = 1
                
                fastest_ema = ema_values[fast_idx, 2]
                metrics["dist_micro_fast"] = (price - fastest_ema) / atr

        # --- 2. BIAS LINE (The 2048 SMMA) ---
        bias_series = close.ewm(alpha=1.0/self.bias_period, adjust=False).mean().iloc[-2:]
        bias_val = bias_series.iloc[-1]
        bias_prev = bias_series.iloc[-0]
        
        metrics["dist_bias"] = (price - bias_val) / atr
        metrics["bias_val"] = bias_val
        metrics["bias_slope"] = (bias_val - bias_prev) / atr
        
        # --- 2.5 PHYSICS ENGINE V3.0 (Efficiency of Flow) ---
        # Mass = Tick Volume (Energy required to move price)
        # Force = Mass * Acceleration
        # Efficiency = Displacement / Effort
        
        # --- 2.5 PHYSICS ENGINE V4.0 (FEAT Force & Absorption) ---
        # [DOCTORAL UPGRADE]
        # Formula: Force = (Body_Size * Volume) / ATR
        # Context: "The Strength Test" (Absorption)
        
        if "volume" in df.columns:
            # 1. VOLUME METRICS FIRST (Needed for FEAT Force)
            current_vol = df["volume"].iloc[-1]
            
            # [MATH SENIOR FULLSTACK - subskill_computational]
            # RVOL: Relative Volume (Vectorized Rolling Mean, No Loops)
            # ε = 1e-9 for numerical stability
            mean_vol = df["volume"].rolling(20).mean().iloc[-1]
            rvol = current_vol / (mean_vol + 1e-9)
            metrics["rvol"] = rvol
            
            # 2. BODY METRICS
            open_p = df["open"].iloc[-1]
            close_p = df["close"].iloc[-1]
            body_size = abs(close_p - open_p)
            
            # Wick Ratio (Efficiency)
            high_p = df["high"].iloc[-1]
            low_p = df["low"].iloc[-1]
            range_size = high_p - low_p
            if range_size == 0: range_size = 1e-9
            
            wick_ratio = (range_size - body_size) / range_size
            
            # 3. [MATH SENIOR FULLSTACK - Kinetic Physicist]
            # ═══════════════════════════════════════════════════════════
            # DOCTORAL FORMULA: Force = (Body / (ATR + ε)) * RVOL
            # - Body/ATR = Normalized Displacement (Dimensionless)
            # - RVOL = Relative Volume (Effort relative to market norm)
            # - Product = True Institutional Force (Comparable across assets)
            # ═══════════════════════════════════════════════════════════
            feat_force = (body_size / (atr + 1e-9)) * rvol
            
            # [MATH SENIOR FULLSTACK - subskill_financial]
            # Volume Efficiency: Body / (Volume * ATR)
            # High Eff = Clean Drive (Institutional). Low Eff = Churn (HFT).
            feat_efficiency = body_size / ((current_vol * atr) + 1e-9)
            
            # Store Metrics
            metrics["feat_force"] = feat_force
            metrics["feat_efficiency"] = feat_efficiency
            metrics["wick_ratio"] = wick_ratio
            
            # 3. ABSORPTION TEST (State Machine)
            abs_result = self.validator.check_absorption_state(df)
            state_map = {"NEUTRAL": 0.0, "IMPULSE": 1.0, "MONITORING": 2.0, "CONFIRMED": 3.0, "FAILED": -1.0}
            
            metrics["absorption_state"] = state_map.get(abs_result["state"], 0.0)
            metrics["absorption_progress"] = float(abs_result["progress"])
            
            # Use calculated FEAT Force if available via validator (or keep local calculation)
            # Validator calculates it for history. Local calc is for current/last.
            # Let's ensure 'feat_force' matches the latest candle if valid.
            if abs_result["state"] == "IMPULSE":
                 metrics["feat_force"] = abs_result["feat_force"]
            
            # 3. ABSORPTION TEST (The 50% Rule)
            # We need to look at the PARENT candle (Impulse) vs recent candles
            # This is complex on a single-row basis. We check "Is current candle absorbed?"
            # Improved Logic: We return a 'strength_score' based on recent history.
            
            # Simple Proxy: If previous candle was BIG, and current is SMALL and INSIDE -> ABSORPTION/CONTINUATION
            # If previous was BIG, and current ENGULFS it -> REVERSAL
            
            prev_body = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
            prev_vol = df["volume"].iloc[-2]
            
            # [KINETIC REPAIR] Use RVOL for Previous Force (Standardized Unit)
            prev_mean_vol = df["volume"].rolling(20).mean().iloc[-2]
            prev_rvol = prev_vol / (prev_mean_vol + 1e-9)
            
            prev_force = (prev_body / (atr + 1e-9)) * prev_rvol
            
            is_inside = (high_p < df["high"].iloc[-2]) and (low_p > df["low"].iloc[-2])
            
            metrics["is_inside_bar"] = 1.0 if is_inside else 0.0
            metrics["prev_force"] = prev_force
            
        else:
            metrics["feat_force"] = 0.0
            metrics["wick_ratio"] = 0.0
            metrics["rvol"] = 0.0
        
        # --- 3. INTER-LAYER RELATIONSHIPS (The "Change in Relation") ---
        # [ELASTICITY PROTOCOL]
        # 1. Fractal Alignment Index (-1.0 to 1.0)
        # Measures how ordered the layers are.
        # +1 = Perfect Bullish Stack (Micro > Struct > Macro)
        # -1 = Perfect Bearish Stack (Micro < Struct < Macro)
        # 0 = Chaos / Entanglement
        
        c_micro = centroids["micro"]
        c_struct = centroids["structure"]
        c_macro = centroids["macro"]
        
        # Bullish Score
        bull_score = 0.0
        if c_micro > c_struct: bull_score += 0.5
        if c_struct > c_macro: bull_score += 0.5
        
        # Bearish Score
        bear_score = 0.0
        if c_micro < c_struct: bear_score += 0.5
        if c_struct < c_macro: bear_score += 0.5
        
        metrics["fractal_alignment_index"] = bull_score - bear_score
        
        # 2. Compression Ratio (0.0 to 1.0)
        # 1.0 = Max Compression (Layers are touching/flat) -> Potential Energy High
        # 0.0 = Max Expansion (Layers far apart) -> Kinetic Energy High
        
        # Width of the ribbon
        ribbon_width = max(c_micro, c_struct, c_macro) - min(c_micro, c_struct, c_macro)
        normalized_width = ribbon_width / (atr + 1e-9)
        
        # We use a Sigmoid-like decay. Historic width ~ 5-10 ATR is expanded. < 1 ATR is compressed.
        # This formula yields high values when width is small.
        metrics["compression_ratio"] = 1.0 / (1.0 + (normalized_width / 2.0))
        
        # 3. Elastic Strain (Hooke's Law)
        # Distance of Price from Macro Centroid (The Mean)
        metrics["elastic_strain"] = (price - c_macro) / (atr + 1e-9)
        
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
        # [SPECTRAL INTEGRATION]
        # Calculate Color Gradient Integrity
        spec_metrics = self.spectral.analyze_spectrum(df["close"], atr=atr)
        metrics.update(spec_metrics)
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
        
        # 5. One-Hot Encoding (Neural Hygiene)
        # Instead of returning an integer ID, we return probability/boolean flags
        
        is_expansion = (pattern_name.startswith("EXPANSION"))
        is_compression = (pattern_name == "COMPRESSION")
        is_reversal = (pattern_name == "DEEP_RETRACEMENT") # or Regime Change
        is_regime_change = (pattern_name.startswith("REGIME_CHANGE"))
        
        return {
            "kinetic_is_expansion": 1.0 if is_expansion else 0.0,
            "kinetic_is_compression": 1.0 if is_compression else 0.0,
            "kinetic_is_reversal": 1.0 if is_reversal else 0.0,
            "kinetic_is_regime_change": 1.0 if is_regime_change else 0.0,
            "kinetic_coherence": float(abs(alignment)),
            "layer_alignment": float(alignment),
            "control_layer_micro": 1.0 if control == "MICRO" else 0.0,
            "control_layer_structure": 1.0 if control == "STRUCTURE" else 0.0,
            "control_layer_macro": 1.0 if control == "MACRO" else 0.0
        }

# Singleton
kinetic_engine = KineticEngine()
