
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, Any
from nexus_core.physics_engine.spectral.deca_core import DecaCoreEngine
from nexus_core.physics_engine.spectral.config import ORDERED_SUBLAYERS

class SpectralTensorBuilder:
    """
    [NEURAL ARCHITECT] Converts Spectral Matrix into High-Density Neural Tensors.
    Adopts Proposal B Adaptive Scaling (Macro-Scale).
    """

    def __init__(self):
        self.engine = DecaCoreEngine()

    def build_tensors(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Main Feature Engineering pipeline.
        Returns: [domino_alignment, elastic_gap, kinetic_whip, bias_regime]
        """
        matrix = self.engine.compute_spectral_matrix(df)
        if matrix.empty: return {}
        
        snapshot = self.engine.get_layer_state(matrix)
        price = df["close"].iloc[-1]
        
        # 1. F - FORMA: DOMINO ALIGNMENT
        # Correlating real rank vs ideal rank [0, 1...9]
        values = [snapshot[sc_id] for sc_id in ORDERED_SUBLAYERS]
        ideal_rank = np.arange(10)
        # For Bullish: SC_1 (Fastest) is Highest price (Rank 0)
        real_rank = np.argsort(np.argsort(values)[::-1]) # Double argsort for rank
        
        corr, _ = spearmanr(ideal_rank, real_rank)
        domino_score = 0.0 if np.isnan(corr) else corr
        
        # 2. E - ESPACIO: ELASTIC GAP (Adaptive Scaling)
        # Using Proposal B 'Macro Scale' normalization
        sc1 = snapshot["SC_1_NOISE"]
        sc4 = snapshot["SC_4_SNIPER"]
        sc10 = snapshot["SC_10_AXIS"]
        
        # Scale is the distance between Entry Zone and the Bedrock
        macro_scale = abs(sc4 - sc10) + 1e-9
        elastic_gap = (sc1 - sc4) / macro_scale

        # 3. A - ACELERACIÓN: KINETIC WHIP
        # Velocity of the Micro-Divergence
        if len(matrix) > 2:
            div_history = matrix["SC_1_NOISE"] - matrix["SC_2_FAST"]
            kinetic_whip = div_history.diff().iloc[-1]
        else:
            kinetic_whip = 0.0

        # 4. T - TIEMPO/SESGO: BIAS REGIME
        bias_regime = 1.0 if price > sc10 else -1.0

        return {
            "domino_alignment": float(domino_score),
            "elastic_gap": float(elastic_gap),
            "kinetic_whip": float(kinetic_whip),
            "bias_regime": float(bias_regime),
            "sc10_axis": float(sc10)
        }

# For backward compatibility with previous verification scripts
class SpectralFeatureProcessor(SpectralTensorBuilder):
    def process_spectral_features(self, df: pd.DataFrame) -> Dict[str, float]:
        # Map build_tensors outputs to legacy names if needed, 
        # but here we'll just update names to match Proposal B suggestions.
        tensors = self.build_tensors(df)
        return {
            "domino_alignment": tensors["domino_alignment"],
            "sniper_proximity": tensors["elastic_gap"], # Renamed for E
            "kinetic_whip": tensors["kinetic_whip"],
            "bias_regime": tensors["bias_regime"],
            "sc10_val": tensors["sc10_axis"]
        }
