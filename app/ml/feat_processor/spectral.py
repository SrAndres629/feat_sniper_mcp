
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from typing import Dict, Any, List
from nexus_core.physics_engine.spectral.deca_core import DecaCoreEngine
from nexus_core.physics_engine.spectral.config import ORDERED_SUBLAYERS
from nexus_core.physics_engine.spectral.wavelet_filter import WaveletPrism
from app.skills.volume_profile import volume_profile
from nexus_core.adaptation_engine import adaptation_engine

class SpectralTensorBuilder:
    """
    [NEURAL ARCHITECT] Converts Spectral Matrix into High-Density Neural Tensors.
    Adopts Proposal B Adaptive Scaling (Macro-Scale).
    """

    def __init__(self):
        self.engine = DecaCoreEngine()
        self.prism = WaveletPrism(wavelet='db4', level=2)
        self.vol_history = [] # For regime classification

    def build_tensors(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Main Feature Engineering pipeline: Binocular Vision (Raw + Wavelet).
        Returns: [domino_alignment, elastic_gap, kinetic_whip, bias_regime, energy_burst, trend_purity, spectral_divergence]
        """
        raw_close = df["close"]
        
        # 1. ADAPTATION LAYER: META-BRAIN CONTROL
        # Logic: Detect regime and adjust math windows
        current_energy = df["tick_volume"].rolling(5).mean().iloc[-1] # Simple volume-based proxy
        self.vol_history.append(current_energy)
        if len(self.vol_history) > 100: self.vol_history.pop(0)
        
        adaptive_config = adaptation_engine.get_config_for_regime(
            current_energy, 
            np.array(self.vol_history)
        )
        
        # Apply Adaptive Parameters
        self.prism.level = adaptive_config["wavelet_level"]
        reg_period = adaptive_config["regress_period"]
        vol_res = adaptive_config["volume_resolution"]

        # 2. QUANTUM PRISM: BINOCULAR PRE-PROCESSING
        # Canal B: Wavelet Price (Pure Trend/Inertia)
        wavelet_series = self.prism.denoise_trend(raw_close.iloc[-100:], full_history=True)
        q_metrics = self.prism.get_quantum_tensors(raw_close)
        
        # 2. MATRIX COMPUTATION: HYBRID FUSION
        # We compute two matrices and fuse them
        raw_matrix = self.engine.compute_spectral_matrix(df)
        if raw_matrix.empty: return {}
        
        df_wav = df.iloc[-100:].copy()
        df_wav["close"] = wavelet_series
        wav_matrix = self.engine.compute_spectral_matrix(df_wav)
        
        # Fusion: Micro (Raw) + Macro (Wavelet)
        hybrid_matrix = raw_matrix.copy()
        macro_scs = ["SC_6_BASE", "SC_7_SESSION", "SC_8_DAY", "SC_9_WEEK", "SC_10_AXIS"]
        for sc in macro_scs:
            hybrid_matrix[sc] = wav_matrix[sc]
            
        snapshot = self.engine.get_layer_state(hybrid_matrix)
        price_raw = raw_close.iloc[-1]
        price_wav = wavelet_series.iloc[-1]
        
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
        
        # spectral_divergence: Slope Divergence (Tactical Judas Swing Detection)
        # Adaptive Regression: period shifts from 3 to 12 based on volatility
        def get_slope(series, p=5):
            y = series.tail(p).values
            x = np.arange(len(y))
            slope, _, _, _, _ = linregress(x, y)
            return slope

        micro_slope = get_slope(hybrid_matrix["SC_1_NOISE"], p=reg_period)
        macro_slope = get_slope(hybrid_matrix["SC_10_AXIS"], p=reg_period)
        spectral_divergence = micro_slope - macro_slope

        # 7. VOLUME VISION (Operation Liquid Density)
        # Adaptive Resolution: 64 in stable, 96 in dead markets for precision
        vol_profile = volume_profile.get_profile(df.tail(vol_res), resolution=vol_res)
        # profile_tensor is normalized 1D (size 64)
        vol_tensor = vol_profile.get("profile_tensor", np.zeros(64))
        vol_shape = vol_profile.get("shape", "Neutral")

        return {
            "domino_alignment": float(domino_score),
            "elastic_gap": float(elastic_gap),
            "kinetic_whip": float(kinetic_whip),
            "bias_regime": float(bias_regime),
            "energy_burst": float(energy_burst),
            "trend_purity": float(trend_purity),
            "spectral_divergence": float(spectral_divergence),
            "sc10_axis": float(sc10),
            "volume_profile_tensor": vol_tensor.tolist() if isinstance(vol_tensor, np.ndarray) else vol_tensor,
            "volume_shape_label": vol_shape,
            "vol_scalar": float(adaptive_config["vol_scalar"]),
            "wavelet_level": int(adaptive_config["wavelet_level"])
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
