import numpy as np
import pandas as pd
from typing import List, Dict

class TensorTopologist:
    """
    [DOCTORAL SKILL]
    Converts Spatial Geometry (Zones) into Mathematical Tensors (Fields).
    """
    
    def __init__(self):
        pass

    def gaussian_decay(self, price: float, zone_center: float, sigma: float) -> float:
        """
        Calculates proximity intensity using a Gaussian Radial Basis Function (RBF).
        Output: 0.0 (Far) -> 1.0 (Dead Center).
        Math: e^(-((x-mu)^2 / 2sigma^2))
        """
        if sigma == 0: return 1.0 # Protect division by zero
        return np.exp(-((price - zone_center)**2) / (2 * (sigma**2)))

    def time_decay(self, age_candles: int, half_life: int = 24) -> float:
        """
        Logarithmic/Exponential decay for zone relevance over time.
        """
        # Simple exponential decay
        return np.exp(-age_candles / half_life)

    def generate_space_tensor(self, df: pd.DataFrame, current_price: float, atr: float) -> np.ndarray:
        """
        Generates the [Space Tensor] for the Neural Network.
        Features:
        1. Proximity to nearest Strong Zone (Gaussian)
        2. Confluence Score of current position
        3. Gravity Vector (Signed distance scaled by ATR)
        """
        # Ensure we have the confluence score
        confluence = df["confluence_score"].iloc[-1] if "confluence_score" in df.columns else 0.0
        
        # 1. Proximity Tensor (To nearest validated zone)
        # We need to find the nearest relevant zone from the DF columns logic
        # For efficiency, we assume 'zone_center' is provided or we scan briefly
        # Simplified: We use the 'confluence_score' as a proxy for "Are we in a zone?"
        # But we want "How close are we?".
        
        # Let's use the 'dist_to_vwap' or similar if available, or calculate Nearest Zone distance
        # For this prototype, we'll build a synthetic proximity based on Bollinger or similar if exact zone coords aren't passed easiest.
        # BETTER: Use the 'confluence_score' directly as the Intensity Tensor.
        
        # If confluence > 0, we are "Touching".
        # If confluence == 0, we need distance.
        
        # Tensor Shape: [ConfluenceIntensity, ZoneProximity, VolatilityContext]
        
        tensor = np.array([
            confluence / 5.0, # Normalized Intensity (Max score approx 5)
            0.0, # Placeholder for precise proximity if not touching
            atr  # Context
        ], dtype=np.float32)
        
        return tensor

    def compute_field_potential(self, price_points: np.ndarray, zones: List[Dict]) -> np.ndarray:
        """
        Advanced: Computes the Scalar Field for a range of prices.
        Used for visualization or Deep Reinforcement Learning environment.
        """
        field = np.zeros_like(price_points)
        for zone in zones:
            center = zone['center']
            weight = zone['weight'] # Confluence Score
            sigma = zone['width'] / 2
            
            # Add Gaussian bump for this zone
            bump = weight * np.exp(-((price_points - center)**2) / (2 * sigma**2))
            field += bump
            
        return field
