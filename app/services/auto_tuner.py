"""
AUTO-TUNING MODULE - Bayesian Hyperparameter Optimization
==========================================================
Uses Optuna to automatically find optimal thresholds for:
- FEAT Chain: Acceleration threshold, Displacement multiplier
- Risk Engine: Confidence minimums, Lot sizing
- Drift Monitor: Z-Score thresholds

Runs weekly on shadow mode data to prevent overfitting.
"""

import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger("feat.auto_tuning")

# Optional Optuna import (not required for production)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("[AUTO-TUNE] Optuna not installed. Run: pip install optuna")


class AutoTuner:
    """
    Bayesian hyperparameter optimizer for FEAT Sniper.
    
    Uses Tree-structured Parzen Estimator (TPE) for efficient
    exploration of the hyperparameter space.
    """
    
    # Default hyperparameter ranges
    PARAM_RANGES = {
        # FEAT Chain thresholds
        "acceleration_threshold": (1.5, 3.0),
        "displacement_multiplier": (1.0, 2.5),
        
        # ML Engine
        "confidence_minimum": (0.55, 0.70),
        "hurst_trending_threshold": (0.50, 0.60),
        "hurst_reverting_threshold": (0.40, 0.50),
        
        # Risk Engine
        "lot_confidence_floor": (0.01, 0.05),
        "trailing_stop_atr": (1.0, 2.5),
        
        # Drift Monitor
        "win_rate_min": (0.40, 0.55),
        "profit_factor_min": (0.8, 1.2),
        "ks_pvalue_threshold": (0.01, 0.10)
    }
    
    CONFIG_FILE = "data/auto_tune_config.json"
    STUDY_NAME = "feat_sniper_optimization"
    
    def __init__(self):
        self.current_params = self._load_config()
        self.study: Optional[Any] = None
        
    def _load_config(self) -> Dict[str, float]:
        """Load current tuned parameters from disk."""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load auto-tune config: {e}")
        
        # Return defaults (mid-range values)
        return {k: (v[0] + v[1]) / 2 for k, v in self.PARAM_RANGES.items()}
    
    def _save_config(self, params: Dict[str, float]) -> None:
        """Persist tuned parameters to disk."""
        os.makedirs(os.path.dirname(self.CONFIG_FILE) or ".", exist_ok=True)
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump({
                "params": params,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0"
            }, f, indent=2)
        logger.info(f"[AUTO-TUNE] Saved config: {self.CONFIG_FILE}")
    
    def create_objective(self, evaluate_fn: Callable[[Dict[str, float]], float]):
        """
        Creates an Optuna objective function.
        
        Args:
            evaluate_fn: Function that takes params dict and returns a score
                         (higher is better, e.g., Sharpe ratio or profit factor)
        """
        def objective(trial):
            params = {}
            for name, (low, high) in self.PARAM_RANGES.items():
                params[name] = trial.suggest_float(name, low, high)
            
            # Evaluate the params on shadow mode data
            score = evaluate_fn(params)
            return score
        
        return objective
    
    def optimize(
        self, 
        evaluate_fn: Callable[[Dict[str, float]], float],
        n_trials: int = 50,
        timeout_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Run Bayesian optimization to find optimal hyperparameters.
        
        Args:
            evaluate_fn: Function that evaluates a param set (returns score)
            n_trials: Maximum optimization iterations
            timeout_minutes: Maximum time to spend optimizing
            
        Returns:
            Dict with best params and optimization stats
        """
        if not OPTUNA_AVAILABLE:
            return {
                "status": "ERROR",
                "message": "Optuna not installed. Run: pip install optuna"
            }
        
        # Create study with TPE sampler (good for hyperparameter optimization)
        self.study = optuna.create_study(
            study_name=self.STUDY_NAME,
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        objective = self.create_objective(evaluate_fn)
        
        logger.info(f"[AUTO-TUNE] Starting optimization: {n_trials} trials, {timeout_minutes}min timeout")
        
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_minutes * 60,
            show_progress_bar=True
        )
        
        # Extract best params
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        # Save to disk
        self._save_config(best_params)
        self.current_params = best_params
        
        result = {
            "status": "SUCCESS",
            "best_score": round(best_score, 4),
            "best_params": {k: round(v, 4) for k, v in best_params.items()},
            "trials_completed": len(self.study.trials),
            "optimization_history": [
                {"trial": t.number, "score": t.value}
                for t in self.study.trials[:10]  # First 10 trials
            ]
        }
        
        logger.info(f"[AUTO-TUNE] Optimization complete. Best score: {best_score:.4f}")
        return result
    
    def get_param(self, name: str) -> float:
        """Get a tuned parameter value."""
        return self.current_params.get(name, self.PARAM_RANGES.get(name, (0.5, 0.5))[0])
    
    def get_all_params(self) -> Dict[str, float]:
        """Get all tuned parameters."""
        return self.current_params.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get auto-tuner status."""
        return {
            "optuna_available": OPTUNA_AVAILABLE,
            "config_file": self.CONFIG_FILE,
            "config_exists": os.path.exists(self.CONFIG_FILE),
            "current_params": self.current_params,
            "param_ranges": self.PARAM_RANGES
        }


# Singleton
auto_tuner = AutoTuner()


# =============================================================================
# EXAMPLE EVALUATION FUNCTION (for shadow mode backtesting)
# =============================================================================

def evaluate_on_shadow_data(params: Dict[str, float]) -> float:
    """
    Example evaluation function for auto-tuning.
    
    In production, this would:
    1. Apply the params to a copy of the strategy
    2. Run backtest on last N shadow mode predictions
    3. Return the Sharpe ratio or profit factor
    
    Returns:
        float: Score (higher is better)
    """
    # Placeholder - in production, this runs actual backtesting
    # For now, return a dummy score based on param reasonableness
    score = 0.0
    
    # Prefer moderate acceleration thresholds
    accel = params.get("acceleration_threshold", 2.0)
    score += 1.0 if 1.8 <= accel <= 2.2 else 0.5
    
    # Prefer higher confidence minimums
    conf = params.get("confidence_minimum", 0.6)
    score += conf * 2
    
    # Add some noise to simulate real evaluation variance
    import random
    score += random.uniform(-0.1, 0.1)
    
    return score


# =============================================================================
# MCP-COMPATIBLE ASYNC WRAPPERS
# =============================================================================

async def get_tuner_status() -> Dict[str, Any]:
    """MCP Tool: Get auto-tuner status."""
    return auto_tuner.get_status()


async def get_tuned_param(name: str) -> Dict[str, Any]:
    """MCP Tool: Get a specific tuned parameter."""
    value = auto_tuner.get_param(name)
    return {"param": name, "value": value}


async def run_optimization(n_trials: int = 50) -> Dict[str, Any]:
    """
    MCP Tool: Run hyperparameter optimization.
    
    WARNING: This is CPU-intensive and should only run during off-hours.
    """
    return auto_tuner.optimize(
        evaluate_fn=evaluate_on_shadow_data,
        n_trials=n_trials,
        timeout_minutes=30
    )
