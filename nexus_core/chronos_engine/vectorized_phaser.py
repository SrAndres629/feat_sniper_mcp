import pandas as pd
import numpy as np
import datetime
import pytz
from typing import Dict
from .phaser import MicroPhase, Intent, VolatilityRegime
from .profiles import ProfileLibrary

class VectorizedGoldPhaser:
    """
    [CHRONOS CORE: VECTORIZED]
    High-Performance temporal mapping for massive backtesting.
    Converts entire DatetimeIndex into Institutional Phases and Profiles.
    """
    
    def __init__(self):
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        self.profiles = ProfileLibrary()

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps a DataFrame with a DatetimeIndex (UTC) to Bolivia Local Time 
        and calculates all temporal features in vectorized fashion.
        """
        if df.index.tz is None:
            # Assume UTC if not specified
            df.index = df.index.tz_localize('UTC')
        
        # 1. Convert to Bolivia Time
        local_times = df.index.tz_convert(self.bolivia_tz)
        hours = local_times.hour
        minutes = local_times.minute
        time_floats = hours + minutes / 60.0
        
        # 2. Phase Mapping (Vectorized using np.select)
        # We define the breakpoints for each phase
        
        # --- BLACKOUTS ---
        # 17:00 - 18:00 Bolivia: DEAD_TIME
        # 18:00 - 19:00 Bolivia: CME_CLOSE
        # 19:00 - 20:00 Bolivia: CME_OPEN
        
        conditions = [
            (time_floats >= 18.0) & (time_floats < 19.0), # SYSTEM_RESET
            (time_floats >= 19.0) & (time_floats < 20.0), # ASIA_OPEN
            (time_floats >= 17.0) & (time_floats < 18.0), # POST_NY_LOW_LIQ
            
            # ASIA
            (time_floats >= 20.0) & (time_floats < 22.0), # ASIA_EARLY
            (time_floats >= 22.0) | (time_floats < 0.0),  # ASIA_SETUP (Note: modulo handling)
            (time_floats >= 0.0) & (time_floats < 1.5),   # ASIA_PRE_LONDON
            
            # LONDON
            (time_floats >= 1.5) & (time_floats < 2.0),   # LONDON_PRE_OPEN
            (time_floats >= 2.0) & (time_floats < 3.0),   # LONDON_RAID
            (time_floats >= 3.0) & (time_floats < 3.5),   # LONDON_REACTION
            (time_floats >= 3.5) & (time_floats < 4.5),   # LONDON_EXPANSION
            (time_floats >= 4.5) & (time_floats < 6.0),   # LONDON_TRANSITION
            
            # EURO LUNCH
            (time_floats >= 6.0) & (time_floats < 7.0),   # EURO_LUNCH
            
            # NY
            (time_floats >= 7.0) & (time_floats < 8.0),   # NY_PRE_INDUCTION
            (time_floats >= 8.0) & (time_floats < 8.5),   # NY_SOFT_OPEN
            (time_floats >= 8.5) & (time_floats < 9.0),   # NY_COMEX_OPEN
            (time_floats >= 9.0) & (time_floats < 9.5),   # NY_CONFIRMATION
            (time_floats >= 9.5) & (time_floats < 10.5),  # NY_EXP_MOMENTUM
            (time_floats >= 10.5) & (time_floats < 11.25),# NY_DISTRIBUTION
            (time_floats >= 11.25) & (time_floats < 12.5),# NY_MIDDAY
            (time_floats >= 12.5) & (time_floats < 13.5), # NY_PRE_CLOSE
            (time_floats >= 13.5) & (time_floats < 15.0), # NY_PM_CLOSE
            (time_floats >= 15.0) & (time_floats < 17.0), # POST_NY_LOW_LIQ_2
        ]
        
        # Handle wrap-around for ASIA_SETUP (starts at 22:00 ends at 00:00)
        # Actually 22:00 to 24:00 is correct.
        
        # We'll use INT values to represent enums for easier Select processing
        # 1: ACCUM, 2: MANIP, 3: EXPAN, 4: DIST
        intent_values = [
            1, 1, 1, # Blackouts -> Accum
            1, 1, 2, # Asia
            1, 2, 2, 3, 4, # London
            1, # Euro
            2, 2, 3, 3, 3, 4, 1, 4, 4, 1 # NY
        ]
        
        # Vol Regime: 1: LOW, 2: HIGH, 3: EXTREME, 4: BLACKOUT
        vol_values = [
            4, 4, 1, # Blackout (CME Close/Open, Pre-Close)
            1, 1, 2, # Asia
            1, 3, 2, 2, 1, # London
            1, # Euro
            2, 2, 3, 2, 2, 2, 1, 2, 3, 1 # NY
        ]

        df['temporal_intent'] = np.select(conditions, intent_values, default=1)
        df['temporal_regime'] = np.select(conditions, vol_values, default=1)
        
        # 3. Expected Volatility (Scalar mapping)
        # Asia: 0.3, NY/London Kill: 0.9, Blackout: 0.0, Lull: 0.4
        vol_expected_map = {
            1: 0.3, # LOW
            2: 0.6, # HIGH
            3: 0.9, # EXTREME
            4: 0.0  # BLACKOUT
        }
        df['expected_volatility'] = df['temporal_regime'].map(vol_expected_map)
        
        return df
