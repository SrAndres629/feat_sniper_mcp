import numpy as np
import pandas as pd

def generate_energy_map(df: pd.DataFrame, bins: int = 50) -> np.ndarray:
    """Generates a Spatial Matrix (Energy Map) from liquidity density (Level 54)."""
    if df.empty or len(df) < bins: return np.zeros((1, bins, bins), dtype=np.float32)
    win = df.tail(bins)
    m, s = win["close"].mean(), win["close"].std()
    if s == 0: s = 1.0
    p_min, p_max = m - (3.0 * s), m + (3.0 * s)
    pr = p_max - p_min
    emap = np.zeros((bins, bins), dtype=np.float32)
    for i, (_, row) in enumerate(win.iterrows()):
        pb = int(((row["close"] - p_min) / (pr + 1e-9)) * (bins - 1))
        pb = max(0, min(bins - 1, pb))
        emap[pb, i] = row["volume"]
    mv = emap.max()
    if mv > 0: emap /= mv
    return emap.reshape(1, bins, bins)
