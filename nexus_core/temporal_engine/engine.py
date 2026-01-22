"""
[MODULE 05 - DOCTORAL TEMPORAL ENGINE]
Unified Temporal Dimension Processing.

Philosophy:
- Time is circular, not linear. The TCN must understand midnight continuity.
- A pip at 9:30 AM EST has 10x the institutional mass of a pip at 3:00 AM.
- The best pattern is garbage if it occurs at the wrong time.

Neural Channels 19-22:
- 19: temporal_sin (hour cycle)
- 20: temporal_cos (hour cycle)
- 21: killzone_intensity (Gaussian institutional pulse)
- 22: session_phase (position in session 0.0-1.0)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import pytz

from app.core.config import settings
from .encoders import (
    encode_hour, encode_minute, encode_day_of_week, 
    encode_fractal_block, killzone_intensity, 
    get_session_name, get_session_weight,
    encode_weekly_phase, get_weekly_feat_phase,
    get_mtf_positions, get_mtf_feat_names
)



logger = logging.getLogger("Nexus.TemporalEngine")


class TemporalEngine:
    """
    [v6.0 - DOCTORAL CHRONOS]
    Unified Temporal Dimension Engine with Sinusoidal Encoding
    and Killzone Intensity × RVOL Anomaly Detection.
    
    Neural Channels:
    - 19: temporal_sin
    - 20: temporal_cos  
    - 21: killzone_intensity
    - 22: session_phase
    """
    
    def __init__(self):
        self.utc_tz = pytz.UTC
        self.ny_tz = pytz.timezone('America/New_York')
        
        # Load config
        self.london_hours = settings.TEMPORAL_LONDON_OPEN
        self.ny_hours = settings.TEMPORAL_NY_OPEN
        self.session_weights = settings.TEMPORAL_SESSION_WEIGHTS
        self.fractal_block = settings.TEMPORAL_FRACTAL_BLOCK_SIZE
        
        logger.info(f"[TemporalEngine] Initialized with NY={self.ny_hours}, London={self.london_hours}")

    def compute_temporal_tensor(self, df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
        """
        [CORE] Computes full temporal tensor for neural consumption.
        Returns DataFrame with temporal columns for channels 19-22.
        
        If time column is missing or not datetime, uses index or synthetic fallback.
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Extract hours from time column or index
        if time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            hours = df[time_col].dt.hour.values
            minutes = df[time_col].dt.minute.values
            dow = df[time_col].dt.dayofweek.values
        elif isinstance(df.index, pd.DatetimeIndex):
            hours = df.index.hour.values
            minutes = df.index.minute.values
            dow = df.index.dayofweek.values
        else:
            # Synthetic fallback: assume NY trading hours spread
            n = len(df)
            hours = np.linspace(9, 16, n).astype(int)  # 9 AM to 4 PM
            minutes = np.zeros(n, dtype=int)
            dow = np.full(n, 2)  # Wednesday
            logger.warning("[TemporalEngine] No time data found, using synthetic hours.")
        
        # =============================================
        # CHANNEL 19-20: SINUSOIDAL HOUR ENCODING
        # =============================================
        # Eliminates 23:59 → 00:00 discontinuity
        hour_radians = 2 * np.pi * (hours / 24.0)
        df["temporal_sin"] = np.sin(hour_radians)
        df["temporal_cos"] = np.cos(hour_radians)
        
        # =============================================
        # CHANNEL 21: KILLZONE INTENSITY (Gaussian)
        # =============================================
        # Peak at session opens, decays with distance
        kz_intensity = np.array([
            killzone_intensity(h, m) for h, m in zip(hours, minutes)
        ])
        df["killzone_intensity"] = kz_intensity
        
        # =============================================
        # CHANNEL 22: SESSION PHASE (0.0 - 1.0)
        # =============================================
        # Position within the current session
        # 0.0 = session start, 0.5 = mid-session (peak), 1.0 = session end
        session_phases = []
        for h in hours:
            session = get_session_name(h)
            if session == "NY_OPEN":
                start, end = self.ny_hours
            elif session in ["LONDON", "LONDON_CLOSE"]:
                start, end = self.london_hours
            else:
                start, end = 0, 24  # Full day for dead zones
            
            duration = end - start
            if duration > 0:
                phase = (h - start) / duration
            else:
                phase = 0.5
            session_phases.append(np.clip(phase, 0.0, 1.0))
        
        df["session_phase"] = session_phases
        
        # =============================================
        # BONUS: Session Weight & IPDA Phase
        # =============================================
        df["session_weight"] = [get_session_weight(h) for h in hours]
        df["session_name"] = [get_session_name(h) for h in hours]
        
        # IPDA Phase encoding (0.0 = Accumulation, 0.33 = Manipulation, 0.66 = Expansion, 1.0 = Distribution)
        ipda_map = {
            "ACCUMULATION": 0.0,
            "MANIPULATION": 0.33,
            "EXPANSION": 0.66,
            "DISTRIBUTION": 1.0
        }
        
        # Simple IPDA based on time (refined version would use FractalTimeTracker)
        def get_ipda_phase(h):
            if 0 <= h < 7:
                return "ACCUMULATION"
            elif 7 <= h < 9:
                return "MANIPULATION"
            elif 9 <= h < 16:
                return "EXPANSION"
            else:
                return "DISTRIBUTION"
        
        df["ipda_phase_name"] = [get_ipda_phase(h) for h in hours]
        df["ipda_phase"] = df["ipda_phase_name"].map(ipda_map).fillna(0.0)
        
        # =============================================
        # TEMPORAL ANOMALY: Intensity × RVOL
        # =============================================
        # If high volume in dead zone = potential institutional sweep
        if "volume" in df.columns:
            vol_mean = df["volume"].rolling(20, min_periods=1).mean()
            rvol = df["volume"] / (vol_mean + 1e-9)
            
            # Anomaly = High RVOL × (1 - Killzone Intensity)
            # High RVOL during dead time = suspicious
            df["temporal_anomaly"] = (rvol * (1.0 - df["killzone_intensity"])).clip(0, 5)
        else:
            df["temporal_anomaly"] = 0.0
        
        # =============================================
        # CHANNELS 23-24: WEEKLY CYCLE ENCODING
        # =============================================
        # Sinusoidal day-of-week for 5-day trading week
        dow_encodings = np.array([encode_day_of_week(d) for d in dow])
        df["dow_sin"] = dow_encodings[:, 0]
        df["dow_cos"] = dow_encodings[:, 1]
        
        # Weekly phase (0.0 = Monday open, 1.0 = Friday close)
        df["weekly_phase"] = [encode_weekly_phase(d, h) for d, h in zip(dow, hours)]
        
        # Weekly FEAT phase name
        df["weekly_feat_phase"] = [get_weekly_feat_phase(d) for d in dow]
        
        # =============================================
        # MTF FRACTAL POSITION ENCODING
        # =============================================
        # Position within each timeframe's candle (1-4) and phase (0.0-1.0)
        # Allows NN to discover patterns like "H4-3 = expansion"
        mtf_data = [get_mtf_positions(h, m) for h, m in zip(hours, minutes)]
        
        # Extract each MTF position and phase
        for tf in ["m1", "m5", "m15", "m30", "h1", "h4"]:
            df[f"{tf}_position"] = [d[f"{tf}_position"] for d in mtf_data]
            df[f"{tf}_phase"] = [d[f"{tf}_phase"] for d in mtf_data]
        
        # D1 position within week (using day of week)
        # 1=Mon (Accum), 2=Tue (Manip), 3=Wed (Expand), 4=Thu (Expand), 5=Fri (Distrib)
        df["d1_position"] = [min(d + 1, 4) if d < 5 else 4 for d in dow]
        df["d1_phase"] = df["weekly_phase"]  # Same as weekly phase
        
        # W1 position within month (approximate - week 1-4)
        # This would need actual date info for precise calculation
        # For now, use a proxy based on day of month if available
        df["w1_position"] = 2  # Default to middle of month
        df["w1_phase"] = 0.5
        
        return df



    def get_live_temporal(self, utc_time: datetime = None) -> Dict[str, float]:
        """
        Returns the current temporal metrics for live trading.
        """
        if utc_time is None:
            utc_time = datetime.now(self.utc_tz)
        
        h = utc_time.hour
        m = utc_time.minute
        dow = utc_time.weekday()
        
        h_sin, h_cos = encode_hour(h + m/60.0)
        kz_int = killzone_intensity(h, m)
        dow_sin, dow_cos = encode_day_of_week(dow)
        weekly_ph = encode_weekly_phase(dow, h + m/60.0)
        weekly_feat = get_weekly_feat_phase(dow)
        
        # Session phase
        session = get_session_name(h)
        if session == "NY_OPEN":
            start, end = self.ny_hours
        elif session in ["LONDON", "LONDON_CLOSE"]:
            start, end = self.london_hours
        else:
            start, end = 0, 24
        phase = (h - start) / max(end - start, 1)
        
        return {
            "temporal_sin": float(h_sin),
            "temporal_cos": float(h_cos),
            "killzone_intensity": float(kz_int),
            "session_phase": float(np.clip(phase, 0, 1)),
            "session_weight": float(get_session_weight(h)),
            "session_name": session,
            "utc_hour": h,
            "day_of_week": dow,
            "dow_sin": float(dow_sin),
            "dow_cos": float(dow_cos),
            "weekly_phase": float(weekly_ph),
            "weekly_feat_phase": weekly_feat,
            # MTF Positions
            **{f"{k}": v for k, v in get_mtf_positions(h, m).items()}
        }



# Singleton
temporal_engine = TemporalEngine()

