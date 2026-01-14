"""
MODULO 2 FASE 14: Statistical Validator
Valida que los resultados del piloto Shadow no son ruido estadístico.

Tests implementados:
1. t-test para significancia de win rate
2. Bootstrap confidence intervals
3. Sharpe ratio validation
4. Minimum sample size check
"""
import logging
import numpy as np
from scipy import stats
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger("feat.stat_validator")

class StatisticalValidator:
    """
    Validador estadístico para resultados del piloto Shadow.
    Confirma que el rendimiento observado es estadísticamente significativo.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results = []
        
    def add_result(self, profit: float, was_win: bool):
        """Add a trade result for analysis."""
        self.results.append({
            "profit": profit,
            "win": was_win,
            "timestamp": datetime.now().isoformat()
        })
    
    def run_validation(self) -> Dict[str, Any]:
        """
        Execute full statistical validation suite.
        """
        if len(self.results) < 30:
            return {
                "status": "INSUFFICIENT_DATA",
                "message": f"Need at least 30 trades, have {len(self.results)}",
                "certified": False
            }
        
        profits = [r["profit"] for r in self.results]
        wins = [1 if r["win"] else 0 for r in self.results]
        
        validation = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(self.results),
            "tests": {}
        }
        
        # Test 1: Win Rate Significance (t-test vs 50%)
        win_rate = np.mean(wins)
        t_stat, p_value = stats.ttest_1samp(wins, 0.5)
        validation["tests"]["win_rate"] = {
            "observed": round(win_rate, 4),
            "null_hypothesis": 0.5,
            "t_statistic": round(t_stat, 3),
            "p_value": round(p_value, 4),
            "significant": p_value < self.alpha and win_rate > 0.5,
            "interpretation": "Better than random" if p_value < self.alpha and win_rate > 0.5 else "Not significant"
        }
        
        # Test 2: Profit Mean test (different from 0)
        mean_profit = np.mean(profits)
        t_stat_profit, p_value_profit = stats.ttest_1samp(profits, 0)
        validation["tests"]["profit_mean"] = {
            "mean_profit": round(mean_profit, 2),
            "t_statistic": round(t_stat_profit, 3),
            "p_value": round(p_value_profit, 4),
            "significant": p_value_profit < self.alpha and mean_profit > 0,
            "interpretation": "Profitable" if p_value_profit < self.alpha and mean_profit > 0 else "Not significant"
        }
        
        # Test 3: Sharpe Ratio
        if np.std(profits) > 0:
            sharpe = np.mean(profits) / np.std(profits) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0
        validation["tests"]["sharpe_ratio"] = {
            "value": round(sharpe, 2),
            "threshold": 1.0,
            "passed": sharpe > 1.0,
            "interpretation": "Institutional grade" if sharpe > 2.0 else ("Acceptable" if sharpe > 1.0 else "Below threshold")
        }
        
        # Test 4: Bootstrap Confidence Interval for Mean Profit
        bootstrap_means = []
        for _ in range(1000):
            sample = np.random.choice(profits, size=len(profits), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        ci_excludes_zero = ci_lower > 0
        
        validation["tests"]["bootstrap_ci"] = {
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            "excludes_zero": ci_excludes_zero,
            "interpretation": "Robust positive expectation" if ci_excludes_zero else "May include zero"
        }
        
        # Overall certification
        passed_tests = sum([
            validation["tests"]["win_rate"]["significant"],
            validation["tests"]["profit_mean"]["significant"],
            validation["tests"]["sharpe_ratio"]["passed"],
            validation["tests"]["bootstrap_ci"]["excludes_zero"]
        ])
        
        validation["certified"] = passed_tests >= 3  # Need at least 3/4 tests
        validation["tests_passed"] = passed_tests
        validation["status"] = "CERTIFIED" if validation["certified"] else "NOT_CERTIFIED"
        validation["recommendation"] = "PROCEED_TO_LIVE" if validation["certified"] else "CONTINUE_SHADOW"
        
        if validation["certified"]:
            logger.info(f"✅ STATISTICAL VALIDATION PASSED ({passed_tests}/4 tests)")
        else:
            logger.warning(f"⚠️ Validation incomplete ({passed_tests}/4 tests)")
        
        return validation

# Singleton
stat_validator = StatisticalValidator()

def test_stat_validator():
    """Test with synthetic data."""
    print("=" * 60)
    print("📈 FEAT SYSTEM - MODULE 2 PHASE 14: STATISTICAL VALIDATION")
    print("=" * 60)
    
    import random
    sv = StatisticalValidator()
    
    # Simulate 50 trades with slight edge
    for _ in range(50):
        is_win = random.random() < 0.55  # 55% win rate
        profit = random.uniform(20, 60) if is_win else random.uniform(-40, -15)
        sv.add_result(profit, is_win)
    
    result = sv.run_validation()
    
    print(f"\n📊 Validation Results:")
    print(f"   Status: {result['status']}")
    print(f"   Tests Passed: {result['tests_passed']}/4")
    print(f"   Certified: {'✅ YES' if result['certified'] else '❌ NO'}")
    
    print(f"\n📋 Test Details:")
    for test_name, test_result in result['tests'].items():
        status = '✅' if test_result.get('significant') or test_result.get('passed') or test_result.get('excludes_zero') else '❌'
        print(f"   {status} {test_name}: {test_result.get('interpretation', 'N/A')}")

if __name__ == "__main__":
    test_stat_validator()
