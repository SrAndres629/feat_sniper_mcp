import numpy as np
import pandas as pd
import pywt
from typing import Tuple, Optional
try:
    from numba import njit
except ImportError:
    # Fallback if numba is not available
    def njit(func):
        return func

class WaveletPrism:
    """
    Quantum Prism: Optimized Multiresolution Analysis.
    Optimized for <10ms latency using localized sliding windows and vectorization.
    """

    def __init__(self, wavelet: str = 'db4', level: int = 2):
        self.wavelet = wavelet
        self.level = level
        self.min_window = 64 
        self._wavelet_obj = pywt.Wavelet(wavelet)

    @staticmethod
    @njit
    def _soft_threshold_numba(coeffs: np.ndarray, threshold: float) -> np.ndarray:
        """Accelerated soft thresholding."""
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)

    def _apply_causal_padding(self, data: np.ndarray) -> np.ndarray:
        pad_size = self.min_window // 2
        return np.pad(data, (pad_size, 0), mode='edge')

    def denoise_trend(self, data: pd.Series, full_history: bool = False) -> pd.Series:
        """
        Optimized Denoising. 
        If full_history=False (Real-time mode), it only computes the tail.
        """
        if len(data) < self.min_window:
            return data.copy()

        raw_values = data.values
        
        if not full_history:
            # Real-time slice: Only compute the last point
            window = raw_values[-self.min_window:]
            return self._denoise_single_window(window)
        
        # Batch Mode: Optimized loop
        output = np.zeros(len(data))
        output[:self.min_window-1] = raw_values[:self.min_window-1]
        
        for i in range(self.min_window, len(data) + 1):
            window = raw_values[i - self.min_window : i]
            output[i-1] = self._denoise_single_window(window)
            
        return pd.Series(output, index=data.index)

    def _denoise_single_window(self, window: np.ndarray) -> float:
        """Helper to denoise a single window and return the last point."""
        padded = self._apply_causal_padding(window)
        coeffs = pywt.wavedec(padded, self._wavelet_obj, level=self.level)
        
        # sigma = median(|d1|) / 0.6745
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thresh = sigma * np.sqrt(2 * np.log(len(window)))
        
        denoised_coeffs = [coeffs[0]]
        for d in coeffs[1:]:
            # We use the internal pywt threshold for compatibility with their structure, 
            # but we could use the numba one if we flattened the lists.
            denoised_coeffs.append(pywt.threshold(d, value=thresh, mode='soft'))
            
        reconstructed = pywt.waverec(denoised_coeffs, self._wavelet_obj)
        return reconstructed[-1]

    def get_quantum_tensors(self, data: pd.Series) -> pd.Series:
        """Fast spectral metrics calculation."""
        if len(data) < self.min_window:
            return pd.Series({'energy_burst_z': 0.0, 'trend_purity_index': 0.0})

        window = data.values[-self.min_window:]
        padded = self._apply_causal_padding(window)
        
        coeffs = pywt.wavedec(padded, self._wavelet_obj, level=1)
        details = coeffs[-1]
        
        energy = np.sum(np.square(details[-5:]))
        approx = coeffs[0]
        approx_e = np.sum(np.square(approx[-5:]))
        
        tpi = approx_e / (approx_e + energy + 1e-9)
        
        return pd.Series({
            'energy_burst_z': float(energy),
            'trend_purity_index': float(tpi)
        })

if __name__ == "__main__":
    import time
    s = pd.Series(np.random.normal(0, 1, 500))
    p = WaveletPrism()
    # Warm up numba
    _ = p.denoise_trend(s.iloc[:70], full_history=True)
    
    start = time.time()
    _ = p.denoise_trend(s, full_history=True)
    print(f"Batch Latency (500 pts): {(time.time() - start)*1000:.2f}ms")
    
    start = time.time()
    _ = p.denoise_trend(s, full_history=False)
    print(f"Real-time Latency (1 pt): {(time.time() - start)*1000:.2f}ms")
