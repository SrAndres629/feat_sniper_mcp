"""
MODULO 5 FASE 14: Deployment Gate (Shadow to Live Transition)
Gate final que controla la transición gradual de Shadow a Live.

Criterios para Live:
1. Piloto Shadow completado (24h+)
2. Validación estadística pasada
3. EV positivo y PF > 1.3
4. No drift detectado
5. Exposición gradual (10% -> 25% -> 50% -> 100%)
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger("feat.deployment_gate")

class DeploymentLevel(Enum):
    """Deployment exposure levels."""
    SHADOW = 0       # Paper trading only
    PILOT_10 = 10    # 10% capital exposure
    PILOT_25 = 25    # 25% capital exposure
    PILOT_50 = 50    # 50% capital exposure
    LIVE_100 = 100   # Full capital exposure

class DeploymentGate:
    """
    Controls the gradual transition from Shadow to Live trading.
    Implements stepped exposure to minimize risk during deployment.
    """
    
    def __init__(self):
        self.current_level = DeploymentLevel.SHADOW
        self.level_history = []
        self.promotion_criteria = {
            DeploymentLevel.SHADOW: {
                "min_trades": 50,
                "min_hours": 24,
                "min_win_rate": 0.50,
                "min_profit_factor": 1.2
            },
            DeploymentLevel.PILOT_10: {
                "min_trades": 20,
                "min_hours": 12,
                "min_win_rate": 0.52,
                "min_profit_factor": 1.3
            },
            DeploymentLevel.PILOT_25: {
                "min_trades": 30,
                "min_hours": 24,
                "min_win_rate": 0.53,
                "min_profit_factor": 1.35
            },
            DeploymentLevel.PILOT_50: {
                "min_trades": 50,
                "min_hours": 48,
                "min_win_rate": 0.54,
                "min_profit_factor": 1.4
            }
        }
        self.level_start_time = datetime.now()
        self.level_trades = 0
        self.level_wins = 0
        self.level_profit = 0.0
        self.level_loss = 0.0
        
    def record_trade(self, is_win: bool, profit: float):
        """Record a trade result at current level."""
        self.level_trades += 1
        if is_win:
            self.level_wins += 1
            self.level_profit += profit
        else:
            self.level_loss += abs(profit)
    
    def get_level_metrics(self) -> Dict[str, Any]:
        """Get metrics for current level."""
        hours_elapsed = (datetime.now() - self.level_start_time).total_seconds() / 3600
        win_rate = self.level_wins / self.level_trades if self.level_trades > 0 else 0
        profit_factor = self.level_profit / self.level_loss if self.level_loss > 0 else float('inf')
        
        return {
            "level": self.current_level.name,
            "exposure_pct": self.current_level.value,
            "trades": self.level_trades,
            "wins": self.level_wins,
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 2),
            "hours_elapsed": round(hours_elapsed, 1),
            "profit": round(self.level_profit, 2),
            "loss": round(self.level_loss, 2)
        }
    
    def check_promotion(self) -> Dict[str, Any]:
        """Check if conditions are met for promotion to next level."""
        if self.current_level == DeploymentLevel.LIVE_100:
            return {"eligible": False, "reason": "Already at max level"}
        
        criteria = self.promotion_criteria.get(self.current_level, {})
        metrics = self.get_level_metrics()
        
        checks = {
            "trades": metrics["trades"] >= criteria.get("min_trades", 0),
            "hours": metrics["hours_elapsed"] >= criteria.get("min_hours", 0),
            "win_rate": metrics["win_rate"] >= criteria.get("min_win_rate", 0),
            "profit_factor": metrics["profit_factor"] >= criteria.get("min_profit_factor", 0)
        }
        
        all_passed = all(checks.values())
        
        return {
            "eligible": all_passed,
            "checks": checks,
            "current_level": self.current_level.name,
            "next_level": self._get_next_level().name if all_passed else None,
            "recommendation": "PROMOTE" if all_passed else "HOLD"
        }
    
    def _get_next_level(self) -> DeploymentLevel:
        """Get the next deployment level."""
        levels = list(DeploymentLevel)
        current_idx = levels.index(self.current_level)
        if current_idx < len(levels) - 1:
            return levels[current_idx + 1]
        return self.current_level
    
    def promote(self) -> Dict[str, Any]:
        """Promote to next level if eligible."""
        check = self.check_promotion()
        
        if not check["eligible"]:
            return {"success": False, "reason": "Not eligible for promotion", "checks": check["checks"]}
        
        old_level = self.current_level
        self.current_level = self._get_next_level()
        
        # Record promotion
        self.level_history.append({
            "from": old_level.name,
            "to": self.current_level.name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_level_metrics()
        })
        
        # Reset level metrics
        self.level_start_time = datetime.now()
        self.level_trades = 0
        self.level_wins = 0
        self.level_profit = 0.0
        self.level_loss = 0.0
        
        logger.info(f"🚀 PROMOTED: {old_level.name} -> {self.current_level.name}")
        
        return {
            "success": True,
            "from": old_level.name,
            "to": self.current_level.name,
            "new_exposure": f"{self.current_level.value}%"
        }
    
    def demote(self, reason: str = "Performance degradation"):
        """Demote to previous level (emergency)."""
        levels = list(DeploymentLevel)
        current_idx = levels.index(self.current_level)
        
        if current_idx > 0:
            old_level = self.current_level
            self.current_level = levels[current_idx - 1]
            logger.warning(f"⚠️ DEMOTED: {old_level.name} -> {self.current_level.name} ({reason})")
    
    def get_lot_multiplier(self) -> float:
        """Get lot size multiplier based on current exposure level."""
        return self.current_level.value / 100.0

# Singleton
deployment_gate = DeploymentGate()

def test_deployment_gate():
    """Test the deployment gate."""
    print("=" * 60)
    print("🚀 FEAT SYSTEM - MODULE 5 PHASE 14: DEPLOYMENT GATE")
    print("=" * 60)
    
    gate = DeploymentGate()
    
    print(f"\n📊 Initial State:")
    print(f"   Level: {gate.current_level.name}")
    print(f"   Exposure: {gate.current_level.value}%")
    
    # Simulate trades to meet criteria
    for i in range(55):
        is_win = i % 2 == 0  # Alternating wins (50% WR)
        profit = 30 if is_win else -25
        gate.record_trade(is_win, profit)
    
    # Force time criteria (for testing)
    gate.level_start_time = datetime.now() - timedelta(hours=25)
    
    metrics = gate.get_level_metrics()
    print(f"\n📈 Current Metrics:")
    print(f"   Trades: {metrics['trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    
    check = gate.check_promotion()
    print(f"\n🎯 Promotion Check:")
    print(f"   Eligible: {'✅ YES' if check['eligible'] else '❌ NO'}")
    for check_name, passed in check['checks'].items():
        print(f"   {'✅' if passed else '❌'} {check_name}")

if __name__ == "__main__":
    test_deployment_gate()
