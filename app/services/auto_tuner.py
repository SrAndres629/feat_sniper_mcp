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

    def optimize_genetic(self, n_gen: int = 5, pop_size: int = 20) -> Dict[str, Any]:
        """
        Runs Genetic Algorithm optimization using DEAP (Distributed Evolutionary Algorithms).
        Evolves 'Survival of the Fittest' params.
        """
        try:
            import random
            import numpy as np
            from deap import base, creator, tools, algorithms
        except ImportError:
            logger.error("DEAP not installed. Run: pip install deap")
            return {"status": "ERROR", "message": "DEAP missing"}

        logger.info(f"[GENETIC] Starting Evolution: {n_gen} Gens, {pop_size} Pop")

        # 1. Setup DEAP (Safe for re-runs)
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
        toolbox = base.Toolbox()
        
        # Genes mapping
        param_names = list(self.PARAM_RANGES.keys())
        
        def random_gene(name):
            low, high = self.PARAM_RANGES[name]
            return random.uniform(low, high)
            
        def init_individual():
            return creator.Individual([random_gene(n) for n in param_names])
            
        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Evaluation Wrapper
        def eval_genome(individual):
            # Clamp values to ranges
            params = {}
            for i, name in enumerate(param_names):
                val = individual[i]
                low, high = self.PARAM_RANGES[name]
                # Mutation might push out of bounds, so clamp
                val = max(low, min(high, val)) 
                params[name] = val
            
            score = evaluate_on_shadow_data(params)
            return (score,) 
            
        toolbox.register("evaluate", eval_genome)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # 2. Run Evolution
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, 
                                       stats=stats, halloffame=hof, verbose=True)
                                       
        # 3. Extract Best
        best_ind = hof[0]
        best_fitness = best_ind.fitness.values[0]
        
        best_params = {}
        for i, name in enumerate(param_names):
            val = best_ind[i]
            low, high = self.PARAM_RANGES[name]
            best_params[name] = max(low, min(high, val))

        # Save to disk
        self._save_config(best_params)
        self.current_params = best_params
        
        logger.info(f"[GENETIC] Evolution Optimized. Fitness: {best_fitness:.4f}")
        logger.info(f"[DNA] Best Genes: {best_params}")
        
        return {
            "status": "SUCCESS",
            "method": "GENETIC_DEAP",
            "best_fitness": best_fitness,
            "best_params": best_params,
            "generations": n_gen
        }


    def select_best_features(self, df: Any, target_col: str = "profit") -> List[str]:
        """
        AutoML: Determine most predictive features using Random Forest importance.
        """
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            
            if not isinstance(df, pd.DataFrame) or df.empty:
                logger.warning("[AUTO-ML] No data provided for feature selection.")
                return []
            
            # Identify feature columns (exclude meta)
            exclude = {"tick_time", "symbol", "label", "profit", "target", "uuid"}
            feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
            
            if not feature_cols:
                return []

            X = df[feature_cols].fillna(0)
            y = df[target_col]
            
            # Train simple RF
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            importances = model.feature_importances_
            
            # Rank
            feature_imp = list(zip(feature_cols, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            
            # Select features with > 1% importance (max 12)
            top_features = [f[0] for f in feature_imp if f[1] > 0.01][:12]
            
            self._save_active_features(top_features)
            logger.info(f"[AUTO-ML] Selected Active Features: {top_features}")
            return top_features
            
        except Exception as e:
            logger.error(f"[AUTO-ML] Feature selection failed: {e}")
            return []

    def _save_active_features(self, features: List[str]):
        path = "data/active_features.json"
        try:
            os.makedirs("data", exist_ok=True)
            with open(path, "w") as f:
                json.dump(features, f)
        except Exception as e:
            logger.error(f"Failed to save features: {e}")


# Singleton
auto_tuner = AutoTuner()


# =============================================================================
# EXAMPLE EVALUATION FUNCTION (for shadow mode backtesting)
# =============================================================================

def evaluate_on_shadow_data(params: Dict[str, float]) -> float:
    """
    Evaluate tuned parameters against shadow mode trade history.
    
    Uses trade_journal to fetch recent shadow predictions and calculates
    theoretical Sharpe ratio if those params had been applied.
    
    Returns:
        float: Score (Sharpe-like metric, higher is better)
    """
    try:
        from app.services.trade_journal import trade_journal
        
        # Fetch recent shadow mode entries
        recent_trades = trade_journal.get_recent_entries(limit=100)
        if not recent_trades or len(recent_trades) < 10:
            # Insufficient data - return heuristic score
            score = 0.0
            accel = params.get("acceleration_threshold", 2.0)
            score += 1.0 if 1.8 <= accel <= 2.2 else 0.5
            conf = params.get("confidence_minimum", 0.6)
            score += conf * 2
            return score
        
        # Calculate theoretical returns with new params
        returns = []
        for trade in recent_trades:
            # Would this trade pass with new params?
            confidence = trade.get("confidence", 0.5)
            accel_score = trade.get("acceleration_score", 0.0)
            
            would_pass = (
                confidence >= params.get("confidence_minimum", 0.6) and
                accel_score >= params.get("acceleration_threshold", 2.0)
            )
            
            if would_pass:
                # Theoretical P&L based on signal accuracy
                actual_profit = trade.get("theoretical_profit", 0.0)
                returns.append(actual_profit)
        
        if not returns:
            return 0.0
        
        # Calculate Sharpe-like ratio
        import numpy as np
        returns_arr = np.array(returns)
        mean_return = np.mean(returns_arr)
        std_return = np.std(returns_arr) if len(returns_arr) > 1 else 1.0
        
        sharpe = mean_return / std_return if std_return > 0 else 0.0
        return float(sharpe)
        
    except Exception as e:
        logger.warning(f"[TUNER] Shadow data evaluation failed: {e}")
        return 0.5  # Neutral score on error


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
    Uses Genetic Algorithms (DEAP) for Level 15 Evolution.
    """
    # Map 'n_trials' to 'pop_size * n_gen' roughly or just use as pop
    return auto_tuner.optimize_genetic(n_gen=5, pop_size=max(10, n_trials))
