import numpy as np
import pandas as pd
from typing import Dict, List
import datetime

class TemporalTensorEngine:
    """
    [CHRONOS ENGINE: LAYER 2]
    Math/Tensor Bridge. Converts discrete Time Engineering into 
    continuous signals for the Neural Network.
    """
    
    def calculate_cyclic_time(self, t: datetime.datetime) -> Dict[str, float]:
        """
        Encodes time of day as Sin/Cos to preserve cyclical nature.
        23:59 should be 'close' to 00:01 in vector space.
        """
        # Minutes from midnight
        minutes_day = t.hour * 60 + t.minute
        max_minutes = 24 * 60
        
        # 2 * pi * (t / T)
        rads = 2 * np.pi * (minutes_day / max_minutes)
        
        return {
            "time_sin": np.sin(rads),
            "time_cos": np.cos(rads)
        }

    def calculate_killzone_intensity(self, t: datetime.datetime) -> float:
        """
        Gaussian Kernel to give continuous 'urgency' score around Kill Zones.
        Centers: 03:00 (London), 09:30 (NY).
        Sigma: Controls the width (approx 1 hour).
        """
        # Convert to fractional hours (e.g., 9:30 -> 9.5)
        h = t.hour + t.minute / 60.0
        
        # Centers for Gaussian Kernels (NY Time)
        # London Open: 3.0
        # NY Open: 9.5
        # London Close: 11.0
        kernels = [3.0, 9.5, 11.0]
        sigma = 1.0 # 1 Hour spread
        
        intensity = 0.0
        for mu in kernels:
            # Gaussian Formula: e^(-(x-mu)^2 / 2sigma^2)
            # Handle circular wrap for early morning if needed? 
            # (Not strictly needed for 3am/9am close centers)
            val = np.exp(-((h - mu)**2) / (2 * sigma**2))
            intensity = max(intensity, val) # Take the strongest signal
            
        return float(intensity)

    def encode_fractal_phase(self, phase: int) -> List[int]:
        """
        One-Hot Encoding for the 4H Fractal Phase (1-4).
        """
        # [Phase 1, Phase 2, Phase 3, Phase 4]
        tensor = [0, 0, 0, 0]
        if 1 <= phase <= 4:
            tensor[phase - 1] = 1
        return tensor
