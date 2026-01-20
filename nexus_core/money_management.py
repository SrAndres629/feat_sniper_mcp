"""
FEAT SNIPER: MONEY MANAGEMENT (RiskOfficer)
===========================================
Professional Fund Management logic.
Implements 'The Vault' (Banking) and Confidence-Based Lot Sizing.
"""

import numpy as np

class MoneyManager:
    def __init__(self, initial_balance: float):
        self.total_balance = initial_balance
        self.vault_balance = 0.0  # Money saved/banked
        self.risk_capital = initial_balance
        self.reinvest_rate = 0.5  # 50% stays in risk capital
        self.vault_rate = 0.5     # 50% goes to the vault

    def update_balance(self, last_trade_profit: float):
        """
        Updates account state after a trade.
        Splits profit between Risk Capital (reinvest) and Vault (bank).
        """
        self.total_balance += last_trade_profit
        
        if last_trade_profit > 0:
            # Reinvest 50%, Send 50% to Vault
            profit_to_vault = last_trade_profit * self.vault_rate
            self.vault_balance += profit_to_vault
        
        # Risk capital is everything EXCEPT the vault money
        self.risk_capital = self.total_balance - self.vault_balance

    def calculate_lot_size(self, stop_loss_pips: int, confidence_score: float, 
                           pip_value: float = 10.0, base_risk_pct: float = 0.01) -> float:
        """
        Dynamic Lot Sizing based on Confidence (Titanium Density).
        
        Formula: Lot_Size = (Risk_Capital * Risk_Pct * Confidence^2) / (SL * PipValue)
        
        Args:
            stop_loss_pips: Distance to SL in pips.
            confidence_score: Value from ConvergenceEngine (0.0 to 1.0).
            pip_value: Value of 1 pip for 1 lot.
            base_risk_pct: Base risk (e.g., 1%).
        """
        if stop_loss_pips <= 0: return 0.01
        
        # Scaling risk by Confidence squared (Confidence-Heavy betting)
        # If confidence is 1.0 -> Risk is 100% of base_risk
        # If confidence is 0.5 -> Risk is 25% of base_risk
        dynamic_risk_pct = base_risk_pct * (confidence_score ** 2)
        
        risk_amount = self.risk_capital * dynamic_risk_pct
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        return round(max(0.01, lot_size), 2)

    def get_fund_status(self) -> dict:
        return {
            "total_equity": self.total_balance,
            "risk_capital": self.risk_capital,
            "vault_balance": self.vault_balance,
            "funding_ratio": self.vault_balance / (self.total_balance + 1e-9)
        }

# Singleton
risk_officer = MoneyManager(1000.0)
