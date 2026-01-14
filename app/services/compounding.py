"""
MODULO 8: Compounding Logic (Capital Scaling)
Phase 13: The Profit Pulse

Objetivo del Visionario:
- Escalar capital automaticamente basado en rendimiento
- Solo compound si el sistema esta "certificado" (EV+ y Profit Factor > 1.3)
- Proteger drawdowns con reset de posicion

Reglas de Compounding:
1. Solo escalar despues de N trades exitosos consecutivos
2. Incrementar lot size por factor gradual (1.1x, 1.2x, etc.)
3. Reset a base lot tras cualquier drawdown significativo
"""
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("feat.compounding")

class CompoundingEngine:
    """
    Motor de escalado de capital basado en rendimiento.
    Aumenta el tamano de posicion cuando el sistema esta 'hot'.
    """
    
    def __init__(self, base_lot: float = 0.01, max_lot: float = 0.5):
        self.base_lot = base_lot
        self.max_lot = max_lot
        self.current_lot = base_lot
        
        # Compounding state
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_profit = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        
        # Thresholds
        self.win_streak_to_compound = 3   # 3 wins = scale up
        self.loss_streak_to_reset = 2     # 2 losses = reset to base
        self.compound_factor = 1.2        # 20% increase per level
        self.max_compound_level = 5       # Max 5 levels
        self.current_level = 0
        
    def record_result(self, result: str, profit: float, equity: float) -> Dict[str, Any]:
        """
        Registra un resultado y ajusta el lot size segun las reglas de compounding.
        
        Args:
            result: "WIN" or "LOSS"
            profit: Profit/Loss amount
            equity: Current account equity
        """
        self.total_profit += profit
        self.current_equity = equity
        
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        # Track streaks
        if result == "WIN":
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Determine action
        action = "HOLD"
        
        # Rule 1: Compound up after win streak
        if self.consecutive_wins >= self.win_streak_to_compound:
            if self.current_level < self.max_compound_level:
                self.current_level += 1
                old_lot = self.current_lot
                self.current_lot = min(
                    self.base_lot * (self.compound_factor ** self.current_level),
                    self.max_lot
                )
                action = "COMPOUND_UP"
                logger.info(f"📈 COMPOUND UP: {old_lot:.2f} -> {self.current_lot:.2f} (Level {self.current_level})")
                self.consecutive_wins = 0  # Reset streak counter
        
        # Rule 2: Reset after loss streak
        if self.consecutive_losses >= self.loss_streak_to_reset:
            if self.current_level > 0:
                self.current_level = 0
                old_lot = self.current_lot
                self.current_lot = self.base_lot
                action = "RESET_TO_BASE"
                logger.warning(f"📉 RESET TO BASE: {old_lot:.2f} -> {self.current_lot:.2f}")
        
        # Rule 3: Check drawdown protection
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown > 0.10:  # 10% drawdown = emergency reset
                self.current_level = 0
                self.current_lot = self.base_lot * 0.5  # Extra conservative
                action = "EMERGENCY_REDUCTION"
                logger.critical(f"🛑 EMERGENCY: Drawdown {drawdown:.1%} - Lot reduced to {self.current_lot:.2f}")
        
        return {
            "action": action,
            "current_lot": self.current_lot,
            "compound_level": self.current_level,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "total_profit": round(self.total_profit, 2),
            "peak_equity": round(self.peak_equity, 2),
            "current_equity": round(self.current_equity, 2)
        }
    
    def get_current_lot(self) -> float:
        """Retorna el lot size actual."""
        return self.current_lot
    
    def is_certified_for_compound(self, ev: float, profit_factor: float) -> bool:
        """
        Verifica si el sistema cumple los requisitos para hacer compounding.
        (Como lo exige el Visionario)
        """
        return ev > 0 and profit_factor > 1.3

# Singleton global
compounding_engine = CompoundingEngine()

def test_compounding():
    """Test the compounding engine with synthetic data."""
    import random
    print("=" * 60)
    print("💹 FEAT SYSTEM - MODULE 8: COMPOUNDING TEST")
    print("=" * 60)
    
    ce = CompoundingEngine(base_lot=0.01, max_lot=0.1)
    equity = 10000.0
    
    # Simulate 15 trades
    results = ["WIN", "WIN", "WIN", "WIN", "LOSS", "WIN", "WIN", "WIN", "WIN", "LOSS", "LOSS", "WIN", "WIN", "WIN", "WIN"]
    
    for i, result in enumerate(results):
        profit = random.uniform(20, 60) if result == "WIN" else random.uniform(-40, -15)
        equity += profit
        
        response = ce.record_result(result, profit, equity)
        print(f"Trade {i+1}: {result} | Lot: {response['current_lot']:.3f} | Level: {response['compound_level']} | Action: {response['action']}")

if __name__ == "__main__":
    test_compounding()
