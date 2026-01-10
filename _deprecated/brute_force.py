"""
brute_force.py - Grid Search & Monte Carlo Optimization for Unified Model
Calibrates FSM thresholds across assets and timeframes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from itertools import product
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from stats_engine import StatsEngine, PercentileStats, load_csv_data, validate_data


@dataclass
class ThresholdSet:
    """FSM threshold configuration."""
    accumulation_compression: float
    expansion_slope: float
    distribution_momentum: float
    reset_speed: float
    hysteresis_margin: float
    min_bars_in_state: int
    
    # FEAT thresholds
    curvature_threshold: float = 0.3
    compression_threshold: float = 0.7
    accel_threshold: float = 0.5
    gap_threshold: float = 1.5


@dataclass
class OptimizationResult:
    """Result from optimization run."""
    thresholds: ThresholdSet
    score: float
    stability: float  # Lower flip-flop rate
    clarity: float    # Confidence scores
    metrics: Dict


class StateSimulator:
    """
    Simulates FSM state transitions for backtesting threshold configurations.
    Mirrors the MQL5 CFSM logic for validation.
    """
    
    ACCUMULATION = 0
    EXPANSION = 1
    DISTRIBUTION = 2
    RESET = 3
    
    STATE_NAMES = ['ACCUMULATION', 'EXPANSION', 'DISTRIBUTION', 'RESET']
    
    def __init__(self, thresholds: ThresholdSet):
        self.thresholds = thresholds
        self.state = self.ACCUMULATION
        self.bars_in_state = 0
        self.transitions = []
        self.confidence_history = []
    
    def classify(self, effort_pct: float, result_pct: float, 
                 compression: float, slope: float, speed: float) -> int:
        """Classify market state based on metrics."""
        
        # RESET: Quick mean reversion
        if abs(speed) > self.thresholds.reset_speed and compression > 0.5:
            return self.RESET
        
        # ACCUMULATION: High effort, low result, compressed
        if (effort_pct > 0.8 and 
            result_pct < 0.2 and 
            compression > self.thresholds.accumulation_compression):
            return self.ACCUMULATION
        
        # EXPANSION: High result, active slope
        if (result_pct > 0.8 and 
            effort_pct < 0.5 and 
            abs(slope) > self.thresholds.expansion_slope):
            return self.EXPANSION
        
        # DISTRIBUTION: Effort up, momentum down
        if (effort_pct > 0.5 and 
            slope < self.thresholds.distribution_momentum and 
            compression < 0.5):
            return self.DISTRIBUTION
        
        return self.ACCUMULATION
    
    def update(self, effort_pct: float, result_pct: float,
               compression: float, slope: float, speed: float,
               bar_index: int) -> Tuple[int, float]:
        """Update state machine with new data. Returns (state, confidence)."""
        
        proposed = self.classify(effort_pct, result_pct, compression, slope, speed)
        confidence = self._calculate_confidence(proposed, effort_pct, result_pct, 
                                                compression, slope)
        
        # Hysteresis
        if proposed != self.state:
            current_conf = self._calculate_confidence(self.state, effort_pct, result_pct,
                                                      compression, slope)
            if confidence < current_conf + self.thresholds.hysteresis_margin * 100:
                proposed = self.state  # Revert
            elif self.bars_in_state >= self.thresholds.min_bars_in_state:
                self.transitions.append({
                    'bar': bar_index,
                    'from': self.STATE_NAMES[self.state],
                    'to': self.STATE_NAMES[proposed],
                    'confidence': confidence
                })
                self.bars_in_state = 0
        
        self.state = proposed
        self.bars_in_state += 1
        self.confidence_history.append(confidence)
        
        return self.state, confidence
    
    def _calculate_confidence(self, state: int, effort_pct: float, result_pct: float,
                              compression: float, slope: float) -> float:
        """Calculate confidence score for a proposed state."""
        base = 50.0
        
        if state == self.ACCUMULATION:
            base += (effort_pct - 0.5) * 30 + (0.5 - result_pct) * 30 + (compression - 0.5) * 40
        elif state == self.EXPANSION:
            base += (result_pct - 0.5) * 40 + abs(slope) * 30
        elif state == self.DISTRIBUTION:
            base += abs(slope) * 40 + (effort_pct - 0.5) * 20
        elif state == self.RESET:
            base += 40  # Reset is always high confidence when triggered
        
        return max(0, min(100, base))
    
    def get_metrics(self) -> Dict:
        """Get simulation metrics."""
        if len(self.confidence_history) == 0:
            return {'transitions': 0, 'avg_confidence': 50, 'stability': 0}
        
        return {
            'transitions': len(self.transitions),
            'avg_confidence': np.mean(self.confidence_history),
            'min_confidence': np.min(self.confidence_history),
            'max_confidence': np.max(self.confidence_history),
            'stability': 1.0 - (len(self.transitions) / max(1, len(self.confidence_history) / 10))
        }


class BruteForceOptimizer:
    """
    Grid search and Monte Carlo optimization for FSM thresholds.
    Tests configurations across historical data to find stable parameters.
    """
    
    def __init__(self, engine: Optional[StatsEngine] = None):
        self.engine = engine or StatsEngine()
        self.results: List[OptimizationResult] = []
    
    def generate_grid(self) -> List[ThresholdSet]:
        """Generate grid of threshold configurations to test."""
        grid = []
        
        # Parameter ranges
        accum_comp = [0.6, 0.7, 0.8]
        expan_slope = [0.2, 0.3, 0.4, 0.5]
        dist_momentum = [-0.1, -0.2, -0.3]
        reset_speed = [1.5, 2.0, 2.5]
        hysteresis = [0.05, 0.1, 0.15]
        min_bars = [2, 3, 4, 5]
        
        for ac, es, dm, rs, hy, mb in product(accum_comp, expan_slope, dist_momentum,
                                               reset_speed, hysteresis, min_bars):
            grid.append(ThresholdSet(
                accumulation_compression=ac,
                expansion_slope=es,
                distribution_momentum=dm,
                reset_speed=rs,
                hysteresis_margin=hy,
                min_bars_in_state=mb
            ))
        
        print(f"[BruteForce] Generated {len(grid)} configurations")
        return grid
    
    def generate_monte_carlo(self, n_samples: int = 100, seed: int = 42) -> List[ThresholdSet]:
        """Generate random threshold configurations via Monte Carlo sampling."""
        np.random.seed(seed)
        configs = []
        
        for _ in range(n_samples):
            configs.append(ThresholdSet(
                accumulation_compression=np.random.uniform(0.5, 0.9),
                expansion_slope=np.random.uniform(0.1, 0.6),
                distribution_momentum=np.random.uniform(-0.4, -0.05),
                reset_speed=np.random.uniform(1.0, 3.0),
                hysteresis_margin=np.random.uniform(0.03, 0.2),
                min_bars_in_state=np.random.randint(2, 6),
                curvature_threshold=np.random.uniform(0.2, 0.5),
                compression_threshold=np.random.uniform(0.5, 0.9),
                accel_threshold=np.random.uniform(0.3, 0.7),
                gap_threshold=np.random.uniform(1.0, 2.5)
            ))
        
        print(f"[BruteForce] Generated {n_samples} Monte Carlo configurations")
        return configs
    
    def evaluate_config(self, thresholds: ThresholdSet,
                        effort: np.ndarray, result: np.ndarray,
                        compression: np.ndarray, slope: np.ndarray,
                        speed: np.ndarray) -> OptimizationResult:
        """Evaluate a single threshold configuration."""
        
        # Convert to percentiles
        effort_pcts = np.array([np.sum(effort < e) / len(effort) for e in effort])
        result_pcts = np.array([np.sum(result < r) / len(result) for r in result])
        
        # Run simulation
        simulator = StateSimulator(thresholds)
        
        for i in range(len(effort)):
            simulator.update(
                effort_pcts[i], result_pcts[i],
                compression[i], slope[i], speed[i], i
            )
        
        metrics = simulator.get_metrics()
        
        # Score: balance stability and clarity
        # High stability (few transitions) + high confidence = good
        stability_score = metrics['stability'] * 40
        clarity_score = (metrics['avg_confidence'] - 50) / 50 * 40
        
        # Penalize extreme transition counts
        trans_penalty = 0
        if metrics['transitions'] < 3:
            trans_penalty = 10  # Too few transitions = not responsive
        elif metrics['transitions'] > len(effort) / 20:
            trans_penalty = 20  # Too many = unstable
        
        score = stability_score + clarity_score - trans_penalty
        
        return OptimizationResult(
            thresholds=thresholds,
            score=score,
            stability=metrics['stability'],
            clarity=metrics['avg_confidence'],
            metrics=metrics
        )
    
    def optimize(self, effort: np.ndarray, result: np.ndarray,
                 compression: np.ndarray, slope: np.ndarray,
                 speed: np.ndarray,
                 method: str = 'grid',
                 n_monte_carlo: int = 100) -> OptimizationResult:
        """
        Run optimization to find best thresholds.
        
        Args:
            effort: Normalized effort array
            result: Normalized result array
            compression: EMA compression values
            slope: EMA slope values
            speed: Price velocity values
            method: 'grid' or 'monte_carlo'
            n_monte_carlo: Samples for Monte Carlo
        
        Returns:
            Best OptimizationResult
        """
        
        if method == 'grid':
            configs = self.generate_grid()
        else:
            configs = self.generate_monte_carlo(n_monte_carlo)
        
        self.results = []
        
        for i, config in enumerate(configs):
            result_obj = self.evaluate_config(config, effort, result, 
                                              compression, slope, speed)
            self.results.append(result_obj)
            
            if (i + 1) % 100 == 0:
                print(f"[BruteForce] Evaluated {i+1}/{len(configs)} configs")
        
        # Sort by score
        self.results.sort(key=lambda x: x.score, reverse=True)
        
        best = self.results[0]
        print(f"\n[BruteForce] Best config:")
        print(f"  Score: {best.score:.2f}")
        print(f"  Stability: {best.stability:.2f}")
        print(f"  Clarity: {best.clarity:.2f}")
        print(f"  Thresholds: {best.thresholds}")
        
        return best
    
    def export_best(self, filepath: str, symbol: str, timeframe: str) -> None:
        """Export best configuration to file."""
        if not self.results:
            print("[BruteForce] No results to export")
            return
        
        best = self.results[0]
        th = best.thresholds
        
        calibration = {
            'symbol': symbol,
            'timeframe': timeframe,
            'version': '1.0.0-optimized',
            'score': best.score,
            
            'effortP20': 0.2,  # These should come from actual data
            'effortP50': 0.5,
            'effortP80': 0.8,
            'resultP20': 0.2,
            'resultP50': 0.5,
            'resultP80': 0.8,
            
            'accumulationCompression': th.accumulation_compression,
            'expansionSlope': th.expansion_slope,
            'distributionMomentum': th.distribution_momentum,
            'resetSpeed': th.reset_speed,
            'hysteresisMargin': th.hysteresis_margin,
            'minBarsInState': th.min_bars_in_state,
            
            'curvatureThreshold': th.curvature_threshold,
            'compressionThreshold': th.compression_threshold,
            'accelThreshold': th.accel_threshold,
            'gapThreshold': th.gap_threshold
        }
        
        # Save as key=value for MQL5
        with open(filepath, 'w') as f:
            for key, value in calibration.items():
                f.write(f"{key}={value}\n")
        
        print(f"[BruteForce] Exported to {filepath}")
    
    def get_top_n(self, n: int = 5) -> List[OptimizationResult]:
        """Get top N results."""
        return self.results[:n]


def run_cross_validation(data_files: List[str], method: str = 'grid') -> Dict[str, OptimizationResult]:
    """
    Run optimization across multiple data files (symbols/timeframes).
    
    Args:
        data_files: List of CSV file paths
        method: Optimization method
    
    Returns:
        Dictionary mapping filename to best result
    """
    results = {}
    
    for filepath in data_files:
        print(f"\n[CrossVal] Processing {filepath}")
        
        try:
            effort, result = load_csv_data(filepath)
            
            # Generate synthetic compression/slope/speed for demo
            # In production, these should come from the CSV
            n = len(effort)
            compression = np.random.uniform(0.3, 0.9, n)
            slope = np.random.normal(0, 0.3, n)
            speed = np.diff(result, prepend=result[0])
            
            optimizer = BruteForceOptimizer()
            best = optimizer.optimize(effort, result, compression, slope, speed, method)
            
            results[filepath] = best
            
        except Exception as e:
            print(f"[CrossVal] Error processing {filepath}: {e}")
    
    return results


if __name__ == "__main__":
    print("Brute Force Optimizer - Unified Model")
    
    # Demo with synthetic data
    np.random.seed(42)
    n = 1000
    
    effort = np.random.lognormal(0, 0.5, n)
    result = np.random.exponential(0.5, n)
    compression = np.random.uniform(0.3, 0.9, n)
    slope = np.random.normal(0, 0.3, n)
    speed = np.diff(result, prepend=result[0])
    
    optimizer = BruteForceOptimizer()
    best = optimizer.optimize(effort, result, compression, slope, speed, 
                              method='monte_carlo', n_monte_carlo=50)
    
    print("\nTop 3 configurations:")
    for i, res in enumerate(optimizer.get_top_n(3)):
        print(f"  {i+1}. Score={res.score:.2f}, Stability={res.stability:.2f}")
