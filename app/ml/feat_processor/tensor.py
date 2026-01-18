import numpy as np
from typing import List, Dict

def tensorize_snapshot(snap: Dict, feature_names: List[str]) -> np.ndarray:
    """Converts a raw snapshot into a Normalized Feature Vector (CUDA-Optimized)."""
    vec = []
    for f in feature_names:
        val = snap.get(f, 0.0)
        try: val = float(val)
        except:
            if isinstance(val, str):
                vu = val.upper()
                if "BOS" in vu or "TREND" in vu: val = 1.0 if any(x in vu for x in ["BULL", "UP"]) else (-1.0 if any(x in vu for x in ["BEAR", "DOWN"]) else 0.0)
                else: val = 0.0
            else: val = 0.0
        # Normalization
        if "score" in f or "rsi" in f: val = val / 100.0 if val > 1.0 else val
        elif "accel" in f: val = val / 10.0 if val > 1.0 else val
        vec.append(float(val))
    vnp = np.array(vec, dtype=np.float32)
    return np.nan_to_num(vnp, nan=0.0, posinf=1.0, neginf=-1.0)
