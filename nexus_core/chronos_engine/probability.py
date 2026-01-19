import datetime
import numpy as np
import pytz
from dataclasses import dataclass

@dataclass
class TimeProbabilities:
    p_manipulation: float # Probability of False Moves / Traps
    p_expansion: float    # Probability of Directional Trends
    p_liquidity: float    # Expected Volume Intensity (0-1)

class ChronosProbabilityEngine:
    """
    [BAYESIAN CHRONOS]
    Models Time as a Set of Probability Distributions.
    Uses Gaussian Kernels (KDE) to estimate regime probabilities.
    
    Formula: P(t) = exp( - (t - mu)^2 / (2 * sigma^2) )
    """
    
    def __init__(self):
        # Reference: Bolivia Time (UTC-4)
        self.bolivia_tz = pytz.timezone('America/La_Paz')
        
        # --- KERNEL DEFINITIONS (Bolivia Time) ---
        # Format: (Mean_Hour, Sigma_Hours)
        
        # 1. MANIPULATION KERNELS (Traps)
        # London Raid (02:00-03:00) -> Peak 02:30
        # Asia Inducement (00:00-01:30) -> Peak 00:45
        # NY Pre-Open (07:00-08:00) -> Peak 07:30
        self.manipulation_kernels = [
            (0.75, 0.5),  # 00:45 (Asia Trap)
            (2.5, 0.4),   # 02:30 (London Raid) - Sharp Sigma
            (7.5, 0.5),   # 07:30 (NY Trap)
        ]
        
        # 2. EXPANSION KERNELS (Trends)
        # London Expansion (03:30-04:30) -> Peak 04:00
        # NY Open (08:30-10:30) -> Peak 09:30
        self.expansion_kernels = [
            (4.0, 0.6),   # 04:00 (London Drive)
            (9.5, 0.8),   # 09:30 (NY Stock Open) - Broad Sigma
        ]
        
        # 3. LIQUIDITY KERNELS (Volume/Energy)
        # Peaks coincident with Opens
        self.liquidity_kernels = [
            (2.5, 0.5),   # London Open
            (9.0, 1.0),   # NY Open (Massive)
            (14.5, 0.5)   # CME Close (Spike)
        ]

    def get_probabilities(self, utc_time: datetime.datetime = None) -> TimeProbabilities:
        if utc_time is None:
            utc_time = datetime.datetime.now(datetime.timezone.utc)
            
        local_time = utc_time.astimezone(self.bolivia_tz)
        
        # Convert time to decimal hours (0.0 - 23.99)
        t = local_time.hour + local_time.minute / 60.0
        
        p_manip = self._calculate_mixture(t, self.manipulation_kernels)
        p_exp = self._calculate_mixture(t, self.expansion_kernels)
        p_liq = self._calculate_mixture(t, self.liquidity_kernels)
        
        return TimeProbabilities(p_manip, p_exp, p_liq)

    def _calculate_mixture(self, t: float, kernels: list) -> float:
        """
        Sum of Gaussians (Mixture Model).
        Returns max probability found (OR Sum? Max is better for distinct regimes).
        """
        prob = 0.0
        for mu, sigma in kernels:
            # Handle Midnight Wrap (e.g. 23:00 vs 01:00)
            # Distance should be minimal across 24h boundary
            dist = abs(t - mu)
            dist = min(dist, 24.0 - dist)
            
            val = np.exp(- (dist**2) / (2 * sigma**2))
            prob = max(prob, val) # Take the dominant regime
            
        return float(prob)
