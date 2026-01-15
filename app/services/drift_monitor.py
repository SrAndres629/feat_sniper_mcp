"""
MODULO 7: Drift Monitor (Model Performance Tracking)
Phase 13: The Profit Pulse + [ALPHA FIX] K-S Regime Detection

Objetivo del Visionario:
- Monitorear el rendimiento del modelo en tiempo real
- Detectar "drift" (degradacion) cuando el modelo empieza a fallar
- [FIX] Detectar cambios de régimen via Kolmogorov-Smirnov test
- Trigger automatico para reentrenamiento o pause

Metricas de Drift:
- Rolling Win Rate (ultimas N trades)
- Profit Factor en tiempo real
- Calibration Error (predicciones vs resultados)
- [NEW] K-S Statistic for regime shift detection
"""
import logging
import json
import numpy as np
from scipy import stats
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger("feat.drift_monitor")

class DriftMonitor:
    """
    Monitor de deriva del modelo.
    Detecta cuando el modelo empieza a degradarse para triggers de reentrenamiento.
    
    [ALPHA FIX] Now includes Kolmogorov-Smirnov test for regime shift detection.
    """
    
    DRIFT_THRESHOLDS = {
        "win_rate_min": 0.45,           # Alerta si cae por debajo de 45%
        "profit_factor_min": 1.0,       # Alerta si cae por debajo de 1.0
        "calibration_error_max": 0.3,   # Alerta si error > 30%
        "consecutive_losses_max": 5,    # Alerta tras 5 pérdidas seguidas
        "ks_pvalue_min": 0.05           # [NEW] Alerta si p-value K-S < 5% (regime shift)
    }
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.recent_trades = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.results = deque(maxlen=window_size)
        self.consecutive_losses = 0
        
        # [ALPHA FIX] Historical baseline for K-S comparison
        self.baseline_returns: Optional[np.ndarray] = None
        self.returns_buffer = deque(maxlen=window_size * 2)  # Recent returns
    
    def detect_regime_shift(self) -> Dict[str, Any]:
        """
        [ALPHA FIX] Uses Kolmogorov-Smirnov test to detect if the return distribution
        has changed compared to the historical baseline.
        
        This catches "silent killers" like correlation breakdowns that Z-scores miss.
        """
        if len(self.returns_buffer) < self.window_size:
            return {"regime_shift": False, "reason": "Insufficient data"}
        
        recent_returns = np.array(list(self.returns_buffer)[-self.window_size:])
        
        # If no baseline, use first window as baseline
        if self.baseline_returns is None:
            if len(self.returns_buffer) >= self.window_size * 2:
                self.baseline_returns = np.array(list(self.returns_buffer)[:self.window_size])
            else:
                return {"regime_shift": False, "reason": "Building baseline"}
        
        # Kolmogorov-Smirnov test: H0 = same distribution
        ks_stat, p_value = stats.ks_2samp(self.baseline_returns, recent_returns)
        
        regime_shift = p_value < self.DRIFT_THRESHOLDS["ks_pvalue_min"]
        
        return {
            "regime_shift": regime_shift,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "interpretation": "DISTRIBUTION_CHANGED" if regime_shift else "STABLE",
            "baseline_size": len(self.baseline_returns) if self.baseline_returns is not None else 0,
            "recent_size": len(recent_returns)
        }
        
    def record_trade(self, predicted_p_win: float, actual_result: str, profit: float):
        """
        Registra un trade completado para analisis de drift.
        
        Args:
            predicted_p_win: Probabilidad predicha por el modelo
            actual_result: "WIN" or "LOSS"
            profit: Profit/Loss en unidades monetarias
        """
        trade = {
            "timestamp": datetime.now().isoformat(),
            "predicted_p_win": predicted_p_win,
            "actual_result": actual_result,
            "profit": profit
        }
        
        self.recent_trades.append(trade)
        self.predictions.append(predicted_p_win)
        self.results.append(1 if actual_result == "WIN" else 0)
        
        # Track consecutive losses
        if actual_result == "LOSS":
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        logger.debug(f"📊 Trade recorded: {actual_result} (Consec Losses: {self.consecutive_losses})")
    
    def check_drift(self) -> Dict[str, Any]:
        """
        Analiza el estado actual del modelo y detecta drift.
        """
        if len(self.recent_trades) < 10:
            return {"status": "INSUFFICIENT_DATA", "drift_detected": False}
        
        trades = list(self.recent_trades)
        
        # Calculate metrics
        wins = sum(1 for t in trades if t["actual_result"] == "WIN")
        losses = len(trades) - wins
        win_rate = wins / len(trades)
        
        gross_profit = sum(t["profit"] for t in trades if t["profit"] > 0)
        gross_loss = abs(sum(t["profit"] for t in trades if t["profit"] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calibration: How well do predictions match outcomes?
        predictions = list(self.predictions)
        results = list(self.results)
        calibration_error = sum(abs(p - r) for p, r in zip(predictions, results)) / len(predictions)
        
        # Drift detection
        alerts = []
        if win_rate < self.DRIFT_THRESHOLDS["win_rate_min"]:
            alerts.append(f"Win Rate {win_rate:.1%} < {self.DRIFT_THRESHOLDS['win_rate_min']:.0%}")
        if profit_factor < self.DRIFT_THRESHOLDS["profit_factor_min"]:
            alerts.append(f"Profit Factor {profit_factor:.2f} < {self.DRIFT_THRESHOLDS['profit_factor_min']}")
        if calibration_error > self.DRIFT_THRESHOLDS["calibration_error_max"]:
            alerts.append(f"Calibration Error {calibration_error:.1%} > {self.DRIFT_THRESHOLDS['calibration_error_max']:.0%}")
        if self.consecutive_losses >= self.DRIFT_THRESHOLDS["consecutive_losses_max"]:
            alerts.append(f"Consecutive Losses {self.consecutive_losses} >= {self.DRIFT_THRESHOLDS['consecutive_losses_max']}")
        
        drift_detected = len(alerts) > 0
        
        result = {
            "status": "DRIFT_DETECTED" if drift_detected else "HEALTHY",
            "drift_detected": drift_detected,
            "alerts": alerts,
            "metrics": {
                "win_rate": round(win_rate, 3),
                "profit_factor": round(profit_factor, 2),
                "calibration_error": round(calibration_error, 3),
                "consecutive_losses": self.consecutive_losses,
                "sample_size": len(trades)
            },
            "recommendation": "PAUSE_AND_RETRAIN" if drift_detected else "CONTINUE"
        }
        
        if drift_detected:
            logger.warning(f"🚨 DRIFT DETECTED: {alerts}")
        else:
            logger.info(f"✅ Model Healthy: WR={win_rate:.1%}, PF={profit_factor:.2f}")
            
        return result
    
    def save_state(self, path: str = "drift_monitor_state.json"):
        """Persiste el estado del monitor."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "trades": list(self.recent_trades),
            "consecutive_losses": self.consecutive_losses
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str = "drift_monitor_state.json"):
        """Carga estado previo del monitor."""
        if Path(path).exists():
            with open(path, 'r') as f:
                state = json.load(f)
                for trade in state.get("trades", []):
                    self.recent_trades.append(trade)
                self.consecutive_losses = state.get("consecutive_losses", 0)

# Singleton global
drift_monitor = DriftMonitor()

def test_drift_monitor():
    """Test the drift monitor with synthetic data."""
    import random
    print("=" * 60)
    print("📉 FEAT SYSTEM - MODULE 7: DRIFT MONITOR TEST")
    print("=" * 60)
    
    dm = DriftMonitor()
    
    # Simulate 20 trades with mixed results
    for i in range(20):
        p_win = random.uniform(0.5, 0.7)
        result = "WIN" if random.random() < 0.55 else "LOSS"
        profit = random.uniform(10, 50) if result == "WIN" else random.uniform(-30, -10)
        dm.record_trade(p_win, result, profit)
    
    check = dm.check_drift()
    print(f"\n📊 Drift Check Results:")
    print(f"   Status: {check['status']}")
    print(f"   Win Rate: {check['metrics']['win_rate']:.1%}")
    print(f"   Profit Factor: {check['metrics']['profit_factor']:.2f}")
    print(f"   Calibration Error: {check['metrics']['calibration_error']:.1%}")
    print(f"   Recommendation: {check['recommendation']}")
    
    if check['alerts']:
        print(f"\n⚠️ Alerts: {check['alerts']}")

if __name__ == "__main__":
    test_drift_monitor()
