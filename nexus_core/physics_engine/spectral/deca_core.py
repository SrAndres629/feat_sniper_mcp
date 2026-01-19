
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .config import DECA_LAYERS, ORDERED_SUBLAYERS

class DecaCoreEngine:
    """
    [PHD LEVEL] DECA-CORE Spectral Filter Bank.
    Implements vectorized computation of 10-layer centroids.
    Optimized for high-speed simulation and live inference.
    """

    def __init__(self):
        self.config = DECA_LAYERS
        self.order = ORDERED_SUBLAYERS

    def compute_spectral_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes OHLC into a 10-column Spectral Matrix (SC_1 to SC_10).
        Full history vectorization.
        """
        if "close" not in df.columns:
            return pd.DataFrame()
            
        centroids = {}
        close = df["close"]

        for _, sublayers in self.config.items():
            for sc_id, periods in sublayers.items():
                # Efficient batch EMA calculation
                emas = pd.concat(
                    [close.ewm(span=p, adjust=False).mean() for p in periods],
                    axis=1
                )
                # Reduce to centroid (Denoising)
                centroids[sc_id] = emas.mean(axis=1)

        return pd.DataFrame(centroids)

    def get_layer_state(self, matrix: pd.DataFrame, idx: int = -1) -> Dict[str, float]:
        """Returns the 10-layer snapshot for a specific index."""
        if matrix.empty: return {}
        return matrix.iloc[idx].to_dict()
