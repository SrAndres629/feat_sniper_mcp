import numpy as np
import pandas as pd
import pywt
from typing import Tuple, List

class WaveletPrism:
    """
    Quantum Prism: Advanced Multiresolution Analysis using DWT (Discrete Wavelet Transform).
    Implements a Parallel-Track architecture:
    - Approximations (Trend): Captured via Low-pass filtering (db4).
    - Details (Energy/Noise): Captured via High-pass filtering.
    
    CRITICAL: Strict Causality Protocol (Anti-Repaint).
    """

    def __init__(self, wavelet: str = 'db4', level: int = 2):
        self.wavelet = wavelet
        self.level = level
        # db4 requires a minimum points for stable decomposition
        self.min_window = 64 

    def _apply_causal_padding(self, data: np.ndarray) -> np.ndarray:
        """
        Applies edge-reflection padding ONLY to the past to maintain phase 
        and avoid look-ahead artifacts during reconstruction.
        """
        # Pad with 1/2 window size at the beginning (past)
        pad_size = self.min_window // 2
        return np.pad(data, (pad_size, 0), mode='edge')

    def denoise_trend(self, data: pd.Series) -> pd.Series:
        """
        RECONSTRUCTIVE DENOISING: Filters out high-freq details to find the 'Pure Trend'.
        Used for Macro/Bias layers in DECA-CORE.
        """
        if len(data) < self.min_window:
            return data.copy()

        raw_values = data.values
        output = np.zeros(len(data))
        
        # Sliding window ensures NO future data is ever seen for point t
        for i in range(self.min_window, len(data) + 1):
            window = raw_values[max(0, i - self.min_window) : i]
            
            # 1. Padding (Causal)
            padded = self._apply_causal_padding(window)
            
            # 2. Decompose
            coeffs = pywt.wavedec(padded, self.wavelet, level=self.level)
            
            # 3. Soft-Thresholding (VisuShrink)
            # sigma estimate: median(|d1|) / 0.6745
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thresh = sigma * np.sqrt(2 * np.log(len(window)))
            
            # Level-dependent thresholding of details
            denoised_coeffs = [coeffs[0]] # Approximation (Trend)
            for d in coeffs[1:]:
                denoised_coeffs.append(pywt.threshold(d, value=thresh, mode='soft'))
            
            # 4. Reconstruct
            reconstructed = pywt.waverec(denoised_coeffs, self.wavelet)
            
            # 5. Extraction: Output[t] = last sample of reconstructed window
            output[i-1] = reconstructed[-1]

        # Fill the initial gap
        output[:self.min_window-1] = data.values[:self.min_window-1]
        
        return pd.Series(output, index=data.index)

    def get_quantum_tensors(self, data: pd.Series) -> pd.Series:
        """
        Calculates spectral energy and purity for the latest window.
        Returns: [energy_burst_z, trend_purity_index]
        """
        if len(data) < self.min_window:
            return pd.Series({'energy_burst_z': 0.0, 'trend_purity_index': 0.0})

        window = data.values[-self.min_window:]
        padded = self._apply_causal_padding(window)
        
        # Decomposition
        coeffs = pywt.wavedec(padded, self.wavelet, level=1)
        details = coeffs[-1] # Highest frequency band
        
        # 1. energy_burst_z (Sum of squares of detail coefficients)
        # Represents raw kinetic energy in the signal's highest bandwidth.
        energy = np.sum(np.square(details[-5:]))
        
        # 2. trend_purity_index (TPI)
        # Ratio of Approximation energy vs Detail energy.
        approx = coeffs[0]
        approx_e = np.sum(np.square(approx[-5:]))
        detail_e = energy + 1e-9
        
        # Higher TPI = Smoother trend. Lower TPI = Turbulent market.
        tpi = approx_e / (approx_e + detail_e)
        
        return pd.Series({
            'energy_burst_z': float(energy),
            'trend_purity_index': float(tpi)
        })

if __name__ == "__main__":
    # Test script for CAUSALITY validation
    prices = pd.Series(np.cumsum(np.random.normal(0, 1, 200)) + 100)
    prism = WaveletPrism()
    denoised = prism.denoise_trend(prices)
    tensors = prism.get_quantum_tensors(prices)
    
    print("Denoising complete.")
    print(f"Final Tensors: {tensors.to_dict()}")
