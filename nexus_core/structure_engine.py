import logging

logger = logging.getLogger("feat.structure")

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
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        
        # 1. Momentum: Large body displacement
        body = df['close'].iloc[-1] - df['open'].iloc[-1]
        is_momentum = abs(body) > (atr * 1.5)
        
        # 2. Accumulation: Compressed range (last 3-5 candles)
        recent_range = (df['high'].iloc[-5:-1].max() - df['low'].iloc[-5:-1].min())
        is_accumulation = recent_range < (atr * 1.2)
        
        # 3. Expansion: Breaking the accumulation zone
        upper_bound = df['high'].iloc[-5:-1].max()
        lower_bound = df['low'].iloc[-5:-1].min()
        
        is_expansion_up = df['close'].iloc[-1] > upper_bound and body > 0
        is_expansion_down = df['close'].iloc[-1] < lower_bound and body < 0
        
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
            "direction": 1 if is_expansion_up else (-1 if is_expansion_down else 0)
        }

class StructureEngine:
    """
    Institutional Structure quantification: Fractals, BOS, CHOCH.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "atr_window": 14,
            "weights": {"F": 0.35, "E": 0.25, "A": 0.20, "T": 0.20}
        }
        self.mae_recognizer = MAE_Pattern_Recognizer()

    def identify_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bill Williams Fractals (5-candle pattern).
        A fractal is confirmed when the middle candle is the highest/lowest of 5.
        """
        # High Fractals
        df['fractal_high'] = (df['high'].shift(2) > df['high'].shift(4)) & \
                             (df['high'].shift(2) > df['high'].shift(3)) & \
                             (df['high'].shift(2) > df['high'].shift(1)) & \
                             (df['high'].shift(2) > df['high'])
        
        # Low Fractals
        df['fractal_low'] = (df['low'].shift(2) < df['low'].shift(4)) & \
                            (df['low'].shift(2) < df['low'].shift(3)) & \
                            (df['low'].shift(2) < df['low'].shift(1)) & \
                            (df['low'].shift(2) < df['low'])
        return df

    def detect_structural_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        BOS (Break of Structure): Trend continuation.
        CHOCH (Change of Character): Trend reversal (First break against trend).
        """
        df = self.identify_fractals(df)
        
        # Track last confirmed fractal levels
        df['last_h_fractal'] = df['high'].where(df['fractal_high']).ffill()
        df['last_l_fractal'] = df['low'].where(df['fractal_low']).ffill()
        
        # BOS: Close exceeds last fractal in trend direction
        df['bos_bull'] = (df['close'] > df['last_h_fractal'].shift(1)) & (df['close'].shift(1) <= df['last_h_fractal'].shift(1))
        df['bos_bear'] = (df['close'] < df['last_l_fractal'].shift(1)) & (df['close'].shift(1) >= df['last_l_fractal'].shift(1))
        
        # CHOCH (Simplified): Cross of EMA 9/21 aligned with a break
        ema9 = df['close'].ewm(span=9).mean()
        ema21 = df['close'].ewm(span=21).mean()
        df['choch_bull'] = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1)) & df['bos_bull']
        df['choch_bear'] = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1)) & df['bos_bear']
        
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
        if mae['phase'] == "EXPANSION": f_score += 0.4
        if df['bos_bull'].iloc[-1] or df['bos_bear'].iloc[-1]: f_score += 0.3
        if df['choch_bull'].iloc[-1] or df['choch_bear'].iloc[-1]: f_score += 0.3
        
        # E-Score (Space) - Proxy using FVG from OHLC
        fvg_bull = (df['low'] > df['high'].shift(2)).iloc[-1]
        fvg_bear = (df['high'] < df['low'].shift(2)).iloc[-1]
        e_score = 0.5 if (fvg_bull or fvg_bear) else 0.2
        
        # A-Score (Acceleration) - Simple volatility proxy
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        curr_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        a_score = min(1.0, curr_range / (atr * 2 + 1e-9))
        
        # T-Score (Time) - Handled externally, but adding dummy for index
        t_score = 0.5 

        w = self.config["weights"]
        feat_val = (w['F'] * f_score + w['E'] * e_score + w['A'] * a_score + w['T'] * t_score)
        
        res = pd.DataFrame(index=df.index)
        res['feat_index'] = round(feat_val * 100, 2)
        res['structure_status'] = mae['status']
        res['is_mae_expansion'] = mae['is_expansion']
        
        return res

structure_engine = StructureEngine()

structure_engine = StructureEngine()
