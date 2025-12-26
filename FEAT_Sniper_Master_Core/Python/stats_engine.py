"""
stats_engine.py - Statistical Engine for Unified Model Calibration
Rolling percentiles, buffers, and validation utilities for Python-side calibration.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class RollingBuffer:
    """Circular buffer for efficient rolling statistics."""
    size: int
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    index: int = 0
    full: bool = False
    
    def __post_init__(self):
        self.data = np.zeros(self.size)
    
    def add(self, value: float) -> None:
        """Add value to buffer."""
        self.data[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.index == 0:
            self.full = True
    
    def get_data(self) -> np.ndarray:
        """Get valid data from buffer."""
        if self.full:
            return self.data.copy()
        return self.data[:self.index].copy()
    
    def is_ready(self) -> bool:
        """Check if buffer has minimum data for statistics."""
        return self.full or self.index >= self.size // 2


@dataclass
class PercentileStats:
    """Percentile statistics container."""
    p20: float = 0.0
    p50: float = 0.0
    p80: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0


class StatsEngine:
    """
    Statistical engine for computing rolling percentiles and calibration metrics.
    Used for brute force calibration of FSM thresholds.
    """
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.effort_buffer = RollingBuffer(buffer_size)
        self.result_buffer = RollingBuffer(buffer_size)
        self.compression_buffer = RollingBuffer(buffer_size)
        self.slope_buffer = RollingBuffer(buffer_size)
        
        self._effort_stats: Optional[PercentileStats] = None
        self._result_stats: Optional[PercentileStats] = None
    
    def add_observation(self, effort: float, result: float, 
                        compression: float = 0.0, slope: float = 0.0) -> None:
        """Add a new observation to all buffers."""
        self.effort_buffer.add(effort)
        self.result_buffer.add(result)
        self.compression_buffer.add(compression)
        self.slope_buffer.add(slope)
        
        # Invalidate cached stats
        self._effort_stats = None
        self._result_stats = None
    
    def is_ready(self) -> bool:
        """Check if engine has enough data for reliable statistics."""
        return self.effort_buffer.is_ready() and self.result_buffer.is_ready()
    
    def compute_percentiles(self, data: np.ndarray) -> PercentileStats:
        """Compute percentile statistics for an array."""
        if len(data) == 0:
            return PercentileStats()
        
        return PercentileStats(
            p20=float(np.percentile(data, 20)),
            p50=float(np.percentile(data, 50)),
            p80=float(np.percentile(data, 80)),
            mean=float(np.mean(data)),
            std=float(np.std(data)) if len(data) > 1 else 0.0,
            min=float(np.min(data)),
            max=float(np.max(data))
        )
    
    def get_effort_stats(self) -> PercentileStats:
        """Get effort percentile statistics."""
        if self._effort_stats is None:
            self._effort_stats = self.compute_percentiles(self.effort_buffer.get_data())
        return self._effort_stats
    
    def get_result_stats(self) -> PercentileStats:
        """Get result percentile statistics."""
        if self._result_stats is None:
            self._result_stats = self.compute_percentiles(self.result_buffer.get_data())
        return self._result_stats
    
    def get_compression_stats(self) -> PercentileStats:
        """Get compression percentile statistics."""
        return self.compute_percentiles(self.compression_buffer.get_data())
    
    def get_slope_stats(self) -> PercentileStats:
        """Get slope percentile statistics."""
        return self.compute_percentiles(self.slope_buffer.get_data())
    
    def get_percentile_value(self, value: float, buffer_name: str) -> float:
        """Get the percentile position of a value within a buffer."""
        buffer_map = {
            'effort': self.effort_buffer,
            'result': self.result_buffer,
            'compression': self.compression_buffer,
            'slope': self.slope_buffer
        }
        
        buffer = buffer_map.get(buffer_name)
        if buffer is None:
            return 0.5
        
        data = buffer.get_data()
        if len(data) == 0:
            return 0.5
        
        return float(np.sum(data < value)) / len(data)
    
    def export_calibration(self, symbol: str, timeframe: str) -> Dict:
        """Export calibration data as dictionary."""
        effort = self.get_effort_stats()
        result = self.get_result_stats()
        compression = self.get_compression_stats()
        slope = self.get_slope_stats()
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'calibrationTime': datetime.now().isoformat(),
            'version': '1.0.0',
            
            # Effort percentiles
            'effortP20': effort.p20,
            'effortP50': effort.p50,
            'effortP80': effort.p80,
            
            # Result percentiles
            'resultP20': result.p20,
            'resultP50': result.p50,
            'resultP80': result.p80,
            
            # Derived thresholds (can be optimized via brute force)
            'accumulationCompression': compression.p80,
            'expansionSlope': slope.p80,
            'distributionMomentum': -slope.p20,
            'resetSpeed': result.p80 * 2,
            
            # FEAT thresholds
            'curvatureThreshold': 0.3,
            'compressionThreshold': compression.p80,
            'accelThreshold': result.p80,
            'gapThreshold': 1.5,
            
            # FSM settings
            'hysteresisMargin': 0.1,
            'minBarsInState': 3
        }
    
    def save_calibration(self, filepath: str, symbol: str, timeframe: str) -> None:
        """Save calibration to file in key=value format for MQL5."""
        calibration = self.export_calibration(symbol, timeframe)
        
        with open(filepath, 'w') as f:
            for key, value in calibration.items():
                f.write(f"{key}={value}\n")
        
        print(f"[StatsEngine] Saved calibration to {filepath}")
    
    def save_calibration_json(self, filepath: str, symbol: str, timeframe: str) -> None:
        """Save calibration to JSON file."""
        calibration = self.export_calibration(symbol, timeframe)
        
        with open(filepath, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"[StatsEngine] Saved JSON calibration to {filepath}")


def load_csv_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load effort/result data from CSV file exported by MQL5.
    Expected format: index,effort,result
    """
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    effort = data[:, 1]
    result = data[:, 2]
    return effort, result


def validate_data(effort: np.ndarray, result: np.ndarray) -> Dict[str, bool]:
    """Validate data for common issues."""
    validations = {
        'no_nan_effort': not np.any(np.isnan(effort)),
        'no_nan_result': not np.any(np.isnan(result)),
        'no_inf_effort': not np.any(np.isinf(effort)),
        'no_inf_result': not np.any(np.isinf(result)),
        'positive_effort': np.all(effort >= 0),
        'positive_result': np.all(result >= 0),
        'sufficient_data': len(effort) >= 100,
        'sufficient_variance_effort': np.std(effort) > 1e-10,
        'sufficient_variance_result': np.std(result) > 1e-10
    }
    return validations


if __name__ == "__main__":
    # Example usage
    print("Stats Engine - Unified Model Calibration")
    
    # Create engine
    engine = StatsEngine(buffer_size=500)
    
    # Simulate data
    np.random.seed(42)
    for _ in range(500):
        effort = np.random.lognormal(0, 0.5)
        result = np.random.exponential(0.5)
        compression = np.random.uniform(0.3, 0.9)
        slope = np.random.normal(0, 0.5)
        engine.add_observation(effort, result, compression, slope)
    
    # Get stats
    print(f"Ready: {engine.is_ready()}")
    print(f"Effort P80: {engine.get_effort_stats().p80:.4f}")
    print(f"Result P80: {engine.get_result_stats().p80:.4f}")
    
    # Export
    calibration = engine.export_calibration("EURUSD", "H1")
    print(f"\nCalibration: {json.dumps(calibration, indent=2)}")
