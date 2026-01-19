
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, Any, List
from nexus_core.physics_engine.spectral.deca_core import DecaCoreEngine
from nexus_core.physics_engine.spectral.config import ORDERED_SUBLAYERS
from nexus_core.physics_engine.spectral.wavelet_filter import WaveletPrism

class SpectralTensorBuilder:
    """
    [NEURAL ARCHITECT] Converts Spectral Matrix into High-Density Neural Tensors.
    Adopts Proposal B Adaptive Scaling (Macro-Scale).
    """

    def __init__(self):
        self.engine = DecaCoreEngine()
        self.prism = WaveletPrism(wavelet='db4', level=2)

    def build_tensors(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Main Feature Engineering pipeline: Binocular Vision (Raw + Wavelet).
        Returns: [domino_alignment, elastic_gap, kinetic_whip, bias_regime, energy_burst, trend_purity, spectral_divergence]
        """
        raw_close = df["close"]
        
        # 1. QUANTUM PRISM: BINOCULAR PRE-PROCESSING
        # Canal B: Wavelet Price (Pure Trend/Inertia)
        wavelet_close = self.prism.denoise_trend(raw_close)
        q_metrics = self.prism.get_quantum_tensors(raw_close)
        
        # 2. MATRIX COMPUTATION: HYBRID FUSION
        # We compute two matrices and fuse them
        raw_matrix = self.engine.compute_spectral_matrix(df)
        if raw_matrix.empty: return {}
        
        df_wav = df.copy()
        df_wav["close"] = wavelet_close
        wav_matrix = self.engine.compute_spectral_matrix(df_wav)
        
        # Fusion: Micro (Raw) + Macro (Wavelet)
        hybrid_matrix = raw_matrix.copy()
        macro_scs = ["SC_6_BASE", "SC_7_SESSION", "SC_8_DAY", "SC_9_WEEK", "SC_10_AXIS"]
        for sc in macro_scs:
            hybrid_matrix[sc] = wav_matrix[sc]
            
        snapshot = self.engine.get_layer_state(hybrid_matrix)
        price_raw = raw_close.iloc[-1]
        price_wav = wavelet_close.iloc[-1]
        
        # 3. F - FORMA: DOMINO ALIGNMENT
        values = [snapshot[sc_id] for sc_id in ORDERED_SUBLAYERS]
        ideal_rank = np.arange(10)
        real_rank = np.argsort(np.argsort(values)[::-1]) 
        
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
        if len(hybrid_matrix) > 2:
            div_history = hybrid_matrix["SC_1_NOISE"] - hybrid_matrix["SC_2_FAST"]
            kinetic_whip = div_history.diff().iloc[-1]
        else:
            kinetic_whip = 0.0

        # 5. T - TIEMPO/SESGO: BIAS REGIME
        bias_regime = 1.0 if price_raw > sc10 else -1.0

        # 6. V3 QUANTUM TENSORS (Binocular Discrepancy & Energy)
        # trend_purity_index: 1.0 = Pure Trend, 0.0 = Pure Noise
        trend_purity = q_metrics['trend_purity_index']
        
        # energy_burst: Raw high-freq kinetic energy
        energy_burst = q_metrics['energy_burst_z']
        
        # spectral_divergence: Gap between raw price and wavelet trend (Normalized by Axis)
        spectral_divergence = (price_raw - price_wav) / (sc10 + 1e-9)

        return {
            "domino_alignment": float(domino_score),
            "elastic_gap": float(elastic_gap),
            "kinetic_whip": float(kinetic_whip),
            "bias_regime": float(bias_regime),
            "energy_burst": float(energy_burst),
            "trend_purity": float(trend_purity),
            "spectral_divergence": float(spectral_divergence),
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
            "sniper_proximity": tensors["elastic_gap"],
            "kinetic_whip": tensors["kinetic_whip"],
            "bias_regime": tensors["bias_regime"],
            "energy_burst": tensors["energy_burst"],
            "trend_purity": tensors["trend_purity"],
            "spectral_divergence": tensors["spectral_divergence"],
            "sc10_val": tensors["sc10_axis"]
        }
