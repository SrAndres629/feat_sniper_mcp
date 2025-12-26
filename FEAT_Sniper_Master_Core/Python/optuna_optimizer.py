"""
optuna_optimizer.py - Bayesian Optimization for FSM Thresholds
Replaces brute_force.py with intelligent hyperparameter search.

Improvements over brute_force.py:
- 30-100x faster (100 trials vs 3,888 grid configs)
- Smarter search with TPE sampler
- Early pruning of bad configs
- Visualization of optimization history
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import json
import os
import logging

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[OptunaOptimizer] optuna not installed. Install with: pip install optuna")

from stats_engine import StatsEngine
from brute_force import ThresholdSet, StateSimulator


# Configure logging
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.WARNING)


@dataclass 
class OptimizationConfig:
    """Configuration for Optuna optimization."""
    n_trials: int = 100
    timeout: Optional[int] = None  # Seconds
    n_startup_trials: int = 10
    n_warmup_steps: int = 5
    seed: int = 42
    show_progress: bool = True
    study_name: str = "fsm_optimization"


class OptunaOptimizer:
    """
    Bayesian optimization for FSM thresholds using Optuna.
    
    Advantages over brute force:
    1. Intelligent sampling with TPE (Tree-structured Parzen Estimator)
    2. Early pruning of unpromising trials
    3. Parallelizable
    4. Built-in visualization
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError("optuna is required. Install with: pip install optuna")
        
        self.config = config or OptimizationConfig()
        self.study: Optional[optuna.Study] = None
        self.best_thresholds: Optional[ThresholdSet] = None
        self.optimization_history: List[Dict] = []
        
        # Data to optimize on (set via set_data())
        self.effort: Optional[np.ndarray] = None
        self.result: Optional[np.ndarray] = None
        self.compression: Optional[np.ndarray] = None
        self.slope: Optional[np.ndarray] = None
        self.speed: Optional[np.ndarray] = None
    
    def set_data(self,
                 effort: np.ndarray,
                 result: np.ndarray,
                 compression: np.ndarray,
                 slope: np.ndarray,
                 speed: np.ndarray) -> None:
        """Set the data to optimize on."""
        self.effort = effort
        self.result = result
        self.compression = compression
        self.slope = slope
        self.speed = speed
        
        # Precompute percentiles for efficiency
        n = len(effort)
        self.effort_pcts = np.array([np.sum(effort < e) / n for e in effort])
        self.result_pcts = np.array([np.sum(result < r) / n for r in result])
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        
        # Sample hyperparameters
        thresholds = ThresholdSet(
            accumulation_compression=trial.suggest_float(
                'accumulation_compression', 0.5, 0.9
            ),
            expansion_slope=trial.suggest_float(
                'expansion_slope', 0.1, 0.6
            ),
            distribution_momentum=trial.suggest_float(
                'distribution_momentum', -0.4, -0.05
            ),
            reset_speed=trial.suggest_float(
                'reset_speed', 1.0, 3.0
            ),
            hysteresis_margin=trial.suggest_float(
                'hysteresis_margin', 0.03, 0.2
            ),
            min_bars_in_state=trial.suggest_int(
                'min_bars_in_state', 2, 6
            ),
            curvature_threshold=trial.suggest_float(
                'curvature_threshold', 0.2, 0.5
            ),
            compression_threshold=trial.suggest_float(
                'compression_threshold', 0.5, 0.9
            ),
            accel_threshold=trial.suggest_float(
                'accel_threshold', 0.3, 0.7
            ),
            gap_threshold=trial.suggest_float(
                'gap_threshold', 1.0, 2.5
            )
        )
        
        # Evaluate
        score = self._evaluate(thresholds)
        
        # Report intermediate value for pruning
        trial.report(score, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return score
    
    def _evaluate(self, thresholds: ThresholdSet) -> float:
        """Evaluate a threshold configuration."""
        
        simulator = StateSimulator(thresholds)
        
        for i in range(len(self.effort)):
            simulator.update(
                self.effort_pcts[i],
                self.result_pcts[i],
                self.compression[i],
                self.slope[i],
                self.speed[i],
                i
            )
        
        metrics = simulator.get_metrics()
        
        # Score: balance stability and clarity
        stability_score = metrics['stability'] * 40
        clarity_score = (metrics['avg_confidence'] - 50) / 50 * 40
        
        # Penalize extreme transition counts
        trans_penalty = 0
        n = len(self.effort)
        if metrics['transitions'] < 3:
            trans_penalty = 10
        elif metrics['transitions'] > n / 20:
            trans_penalty = 20
        
        score = stability_score + clarity_score - trans_penalty
        
        return score
    
    def optimize(self) -> ThresholdSet:
        """
        Run Bayesian optimization.
        
        Returns:
            Best ThresholdSet found
        """
        if self.effort is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Create study
        sampler = TPESampler(
            seed=self.config.seed,
            n_startup_trials=self.config.n_startup_trials
        )
        
        pruner = MedianPruner(
            n_startup_trials=self.config.n_startup_trials,
            n_warmup_steps=self.config.n_warmup_steps
        )
        
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        self.study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=self.config.show_progress
        )
        
        # Extract best thresholds
        best_params = self.study.best_params
        self.best_thresholds = ThresholdSet(**best_params)
        
        # Store history
        self.optimization_history = [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in self.study.trials
        ]
        
        return self.best_thresholds
    
    def get_results(self) -> Dict:
        """Get optimization results."""
        if self.study is None:
            return {}
        
        return {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'n_pruned': len([t for t in self.study.trials 
                            if t.state == optuna.trial.TrialState.PRUNED]),
            'optimization_history': self.optimization_history[:10]  # First 10
        }
    
    def export_calibration(self, filepath: str, symbol: str, timeframe: str) -> None:
        """Export best configuration to file for MQL5."""
        if self.best_thresholds is None:
            raise ValueError("No optimization results. Call optimize() first.")
        
        th = self.best_thresholds
        
        calibration = {
            'symbol': symbol,
            'timeframe': timeframe,
            'version': '2.0.0-optuna',
            'score': self.study.best_value if self.study else 0,
            'method': 'optuna_tpe',
            'n_trials': len(self.study.trials) if self.study else 0,
            
            # Percentiles (should come from actual data)
            'effortP20': 0.2,
            'effortP50': 0.5,
            'effortP80': 0.8,
            'resultP20': 0.2,
            'resultP50': 0.5,
            'resultP80': 0.8,
            
            # Optimized thresholds
            'accumulationCompression': th.accumulation_compression,
            'expansionSlope': th.expansion_slope,
            'distributionMomentum': th.distribution_momentum,
            'resetSpeed': th.reset_speed,
            'hysteresisMargin': th.hysteresis_margin,
            'minBarsInState': th.min_bars_in_state,
            
            # FEAT thresholds
            'curvatureThreshold': th.curvature_threshold,
            'compressionThreshold': th.compression_threshold,
            'accelThreshold': th.accel_threshold,
            'gapThreshold': th.gap_threshold
        }
        
        # Save as key=value for MQL5
        with open(filepath, 'w') as f:
            for key, value in calibration.items():
                f.write(f"{key}={value}\n")
        
        # Also save as JSON
        json_path = filepath.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"[OptunaOptimizer] Exported to {filepath}")
    
    def plot_optimization_history(self, filepath: Optional[str] = None):
        """Plot optimization history (requires plotly)."""
        if self.study is None:
            print("[OptunaOptimizer] No study to plot")
            return
        
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self.study)
            
            if filepath:
                fig.write_html(filepath)
                print(f"[OptunaOptimizer] Saved plot to {filepath}")
            else:
                fig.show()
        except ImportError:
            print("[OptunaOptimizer] plotly required for visualization")
    
    def plot_param_importances(self, filepath: Optional[str] = None):
        """Plot parameter importances."""
        if self.study is None:
            print("[OptunaOptimizer] No study to plot")
            return
        
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self.study)
            
            if filepath:
                fig.write_html(filepath)
                print(f"[OptunaOptimizer] Saved plot to {filepath}")
            else:
                fig.show()
        except ImportError:
            print("[OptunaOptimizer] plotly required for visualization")


def run_quick_optimization(effort: np.ndarray,
                           result: np.ndarray,
                           compression: np.ndarray,
                           slope: np.ndarray,
                           speed: np.ndarray,
                           n_trials: int = 50) -> ThresholdSet:
    """
    Quick optimization with default settings.
    
    Args:
        effort: Volume effort array
        result: Price result array
        compression: EMA compression array
        slope: EMA slope array
        speed: Price velocity array
        n_trials: Number of trials
    
    Returns:
        Best ThresholdSet
    """
    config = OptimizationConfig(
        n_trials=n_trials,
        show_progress=True
    )
    
    optimizer = OptunaOptimizer(config)
    optimizer.set_data(effort, result, compression, slope, speed)
    
    best = optimizer.optimize()
    results = optimizer.get_results()
    
    print(f"\n[OptunaOptimizer] Optimization Complete")
    print(f"  Best Score: {results['best_value']:.2f}")
    print(f"  Trials: {results['n_trials']} ({results['n_pruned']} pruned)")
    print(f"\n  Best Thresholds:")
    for key, value in results['best_params'].items():
        print(f"    {key}: {value:.4f}")
    
    return best


if __name__ == "__main__":
    print("Optuna Optimizer - Bayesian Search for FSM Thresholds")
    print("="*60)
    
    if not OPTUNA_AVAILABLE:
        print("Please install optuna: pip install optuna")
        exit(1)
    
    # Demo with synthetic data
    np.random.seed(42)
    n = 1000
    
    effort = np.random.lognormal(0, 0.5, n)
    result = np.random.exponential(0.5, n)
    compression = np.random.uniform(0.3, 0.9, n)
    slope = np.random.normal(0, 0.3, n)
    speed = np.diff(result, prepend=result[0])
    
    # Run optimization
    best_thresholds = run_quick_optimization(
        effort, result, compression, slope, speed,
        n_trials=50
    )
    
    # Export
    output_path = os.path.join(os.path.dirname(__file__), "optuna_calibration.txt")
    
    optimizer = OptunaOptimizer()
    optimizer.set_data(effort, result, compression, slope, speed)
    optimizer.optimize()
    optimizer.export_calibration(output_path, "EURUSD", "H1")
