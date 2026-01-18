import time
import logging
from collections import deque
from typing import Dict, Optional, Tuple, Deque

logger = logging.getLogger("SystemGuard.Resources")

class ResourcePredictor:
    def __init__(self, window_size: int = 60, sample_interval_sec: float = 1.0):
        self.window_size = window_size
        self.sample_interval = sample_interval_sec
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=window_size)
        self._last_sample_time: float = 0.0

    def sample(self) -> Optional[float]:
        now = time.time()
        if now - self._last_sample_time < self.sample_interval: return None
        self._last_sample_time = now
        try:
            import psutil
            ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            self._samples.append((now, ram_mb))
            return ram_mb
        except: return None

    def get_slope(self) -> Optional[float]:
        if len(self._samples) < 10: return None
        n = len(self._samples)
        sx = sum(s[0] for s in self._samples)
        sy = sum(s[1] for s in self._samples)
        sxy = sum(s[0] * s[1] for s in self._samples)
        sxx = sum(s[0] ** 2 for s in self._samples)
        den = n * sxx - sx ** 2
        return (n * sxy - sx * sy) / den if den != 0 else 0.0

    def predict_oom(self, threshold_mb: float = 1000.0, critical_slope_mb_s: float = 1.0) -> Dict:
        if not self._samples: return {"status": "UNKNOWN"}
        curr = self._samples[-1][1]
        slope = self.get_slope()
        res = {"status": "OK", "current_ram_mb": round(curr, 2), "slope_mb_s": round(slope, 6) if slope is not None else None}
        if slope is None: return {**res, "status": "INSUFFICIENT_DATA"}
        if curr > threshold_mb:
            res.update({"status": "CRITICAL", "message": f"RAM {curr:.0f}MB > {threshold_mb:.0f}MB"})
        elif slope > critical_slope_mb_s:
            rem = threshold_mb - curr
            t_oom = rem / slope if slope > 0 else float('inf')
            res.update({"status": "WARNING", "message": f"Leak detected. OOM in ~{t_oom:.0f}s", "time_to_oom_s": round(t_oom, 1)})
        return res
