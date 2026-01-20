import numpy as np
import pandas as pd
import pytz
from typing import Dict
from nexus_core.chronos_engine.vectorized_phaser import VectorizedGoldPhaser

class VectorizedChronosProcessor:
    """
    [CHRONOS CORE: MASSIVE BACKTESTING]
    Vectorized processor for generating temporal features.
    Processes entire DataFrames in milliseconds.
    """
    
    def __init__(self):
        self.phaser = VectorizedGoldPhaser()
        self.ny_tz = pytz.timezone('America/New_York')

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all temporal tensors for the given DataFrame.
        Expected index: DatetimeIndex (UTC).
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        # 0. Prep Time Indices
        # [FIX] Robust index handling: ensure we have a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df.index = pd.to_datetime(df['time'])
            else:
                # Fallback: if no time column, we cannot calculate temporal features
                # but we must prevent the crash. We'll use a dummy UTC index starting now.
                df.index = pd.date_range(start=pd.Timestamp.now(tz='UTC'), periods=len(df), freq='min')

        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
            
        # [CRITICAL FIX] Deduplicate Index
        # Timestamp collisions cause downstream crashes (pd.concat/reindex)
        df = df[~df.index.duplicated(keep='last')]
        
        # [DEFENSE] Sort index just in case
        df = df.sort_index()
            
        # 1. CYCLICAL ENCODING (Doctoral Standard)
        # We use NY time for cycle centers to match Institutional clocks
        ny_times = df.index.tz_convert(self.ny_tz)
        minutes_day = ny_times.hour * 60 + ny_times.minute
        max_minutes = 24 * 60
        rads = 2 * np.pi * (minutes_day / max_minutes)
        
        df['time_sin'] = np.sin(rads)
        df['time_cos'] = np.cos(rads)
        
        # 2. GAUSSIAN KILLZONES (Intensity)
        h_floats = ny_times.hour + ny_times.minute / 60.0
        centers = [3.0, 9.5, 11.0] # London, NY, London Close
        sigma = 1.0 # 1 Hour width
        
        # Vectorized Gaussian Calculation
        intensities = []
        for mu in centers:
            # Handle circular wrap-around (important for 24h cycles)
            # Distance d = |h - mu|. We want min(d, 24 - d)
            diff = np.abs(h_floats - mu)
            dist = np.minimum(diff, 24 - diff)
            val = np.exp(-(dist**2) / (2 * sigma**2))
            intensities.append(val)
            
        df['killzone_intensity'] = np.max(intensities, axis=0)

        # 3. FRACTAL PHASES (4H Blocks)
        # Phase = (Hour % 4) + 1
        fractal_phases = (ny_times.hour % 4) + 1
        for i in range(1, 5):
            df[f'fractal_phase_{i}'] = (fractal_phases == i).astype(float)

        # 4. SESSION LOGIC (Intents & Profiles)
        df = self.phaser.process_dataframe(df)
        
        # 5. ONE-HOT INTENTS
        # 1: ACCUM, 2: MANIP, 3: EXPAN, 4: DIST
        for i, name in enumerate(['accum', 'manip', 'expan', 'dist'], 1):
            df[f'intent_{name}'] = (df['temporal_intent'] == i).astype(float)
            
        # 6. WEEKDAY ENCODING
        weekdays = df.index.weekday # 0-6
        for i in range(7):
            df[f'day_{i}'] = (weekdays == i).astype(float)
            
        return df
