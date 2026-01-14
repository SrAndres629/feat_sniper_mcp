"""
MODULO 3 FASE 14: Risk Parity Position Sizing
Position sizing basado en volatilidad del activo.

Principios:
1. Cada trade asume un riesgo constante (ej: 1% del capital)
2. El lot size se ajusta inversamente proporcional a la volatilidad
3. Mayor volatilidad = menor position size
"""
import logging
from typing import Dict, Any

logger = logging.getLogger("feat.risk_parity")

class RiskParityCalculator:
    """
    Calculator for risk-parity based position sizing.
    Adjusts lot size based on current volatility to maintain constant risk.
    """
    
    def __init__(self, base_risk_percent: float = 1.0, account_balance: float = 10000.0):
        """
        Initialize the risk parity calculator.
        
        Args:
            base_risk_percent: Percentage of account to risk per trade (default 1%)
            account_balance: Current account balance
        """
        self.base_risk_percent = base_risk_percent
        self.account_balance = account_balance
        self.base_atr = 0.0  # Historical average ATR for the symbol
        
    def update_balance(self, new_balance: float):
        """Update account balance for position sizing."""
        self.account_balance = new_balance
        
    def set_base_atr(self, atr: float):
        """Set the baseline ATR for volatility comparison."""
        self.base_atr = atr
        
    def calculate_lot_size(self, 
                           current_atr: float, 
                           stop_loss_pips: float,
                           pip_value: float = 10.0) -> Dict[str, Any]:
        """
        Calculate optimal lot size based on risk parity principles.
        
        Args:
            current_atr: Current ATR value
            stop_loss_pips: Distance to stop loss in pips
            pip_value: Value per pip per standard lot (default $10 for XAUUSD)
            
        Returns:
            Dictionary with lot size and calculation details
        """
        # Calculate dollar risk
        dollar_risk = self.account_balance * (self.base_risk_percent / 100)
        
        # Volatility adjustment factor
        if self.base_atr > 0 and current_atr > 0:
            vol_factor = self.base_atr / current_atr
            vol_factor = max(0.5, min(1.5, vol_factor))  # Cap between 0.5x and 1.5x
        else:
            vol_factor = 1.0
        
        # Calculate base lot from risk
        if stop_loss_pips > 0:
            base_lot = dollar_risk / (stop_loss_pips * pip_value)
        else:
            base_lot = 0.01
        
        # Apply volatility adjustment
        adjusted_lot = base_lot * vol_factor
        
        # Round to standard lot increments
        adjusted_lot = round(adjusted_lot, 2)
        adjusted_lot = max(0.01, min(1.0, adjusted_lot))  # Min 0.01, max 1.0
        
        result = {
            "lot_size": adjusted_lot,
            "base_lot": round(base_lot, 3),
            "vol_factor": round(vol_factor, 2),
            "dollar_risk": round(dollar_risk, 2),
            "current_atr": current_atr,
            "base_atr": self.base_atr,
            "risk_percent": self.base_risk_percent
        }
        
        logger.debug(f"📊 Risk Parity: Lot={adjusted_lot}, VolFactor={vol_factor:.2f}")
        return result
    
    def get_regime_multiplier(self, regime: str) -> float:
        """
        Get risk multiplier based on market regime.
        """
        multipliers = {
            "LAMINAR": 1.0,      # Normal conditions
            "TURBULENT": 0.5,   # Reduce risk
            "CHAOS": 0.0,       # No trading
            "OVERLAP": 1.2     # London/NY overlap - higher edge
        }
        return multipliers.get(regime, 1.0)

# Singleton
risk_parity = RiskParityCalculator()

def test_risk_parity():
    """Test risk parity calculations."""
    print("=" * 60)
    print("⚖️ FEAT SYSTEM - MODULE 3 PHASE 14: RISK PARITY")
    print("=" * 60)
    
    rp = RiskParityCalculator(base_risk_percent=1.0, account_balance=10000)
    rp.set_base_atr(15.0)  # Baseline ATR of 15 pips
    
    scenarios = [
        {"atr": 10.0, "sl": 20, "desc": "Low volatility"},
        {"atr": 15.0, "sl": 20, "desc": "Normal volatility"},
        {"atr": 25.0, "sl": 20, "desc": "High volatility"},
    ]
    
    print(f"\n📊 Position Sizing Scenarios (Account: ${rp.account_balance}, Risk: {rp.base_risk_percent}%)")
    print("-" * 50)
    
    for scenario in scenarios:
        result = rp.calculate_lot_size(scenario["atr"], scenario["sl"])
        print(f"\n{scenario['desc']} (ATR={scenario['atr']}):")
        print(f"   Lot Size: {result['lot_size']}")
        print(f"   Vol Factor: {result['vol_factor']}x")
        print(f"   Dollar Risk: ${result['dollar_risk']}")

if __name__ == "__main__":
    test_risk_parity()
