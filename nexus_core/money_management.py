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
        
        # Growth Phases Config
        self.PHASE_1_LIMIT = 100.0   # Survival Phase (<$100)
        self.PHASE_2_LIMIT = 500.0   # Consolidation Phase (<$500)
        
        # Initial State Update
        self._update_growth_phase()

    def _update_growth_phase(self):
        """Determines the Vault Rate and Risk ceiling based on account size."""
        if self.total_balance < self.PHASE_1_LIMIT:
            # PHASE 1: SURVIVAL & SNOWBALL ($20 -> $100)
            # Goal: Grow fast out of the danger zone.
            self.vault_rate = 0.0      # Reinvest 100% of profits
            self.max_risk_pct = 0.05   # Allow up to 5% risk on Titanium signals
            self.phase_name = "SURVIVAL (Aggressive)"
            
        elif self.total_balance < self.PHASE_2_LIMIT:
            # PHASE 2: CONSOLIDATION ($100 -> $500)
            # Goal: Balance growth with safety.
            self.vault_rate = 0.20     # Bank 20% of profits
            self.max_risk_pct = 0.03   # Max 3% risk
            self.phase_name = "CONSOLIDATION (Moderate)"
            
        else:
            # PHASE 3: INSTITUTIONAL (>$500)
            # Goal: Wealth Preservation and Income.
            self.vault_rate = 0.50     # Bank 50% of profits (The Vault Standard)
            self.max_risk_pct = 0.015  # Max 1.5% risk
            self.phase_name = "INSTITUTIONAL (Conservative)"

    def update_balance(self, last_trade_profit: float):
        """
        Updates account state and re-evaluates growth phase.
        """
        self.total_balance += last_trade_profit
        
        if last_trade_profit > 0:
            # Tax the profit for the Vault based on current phase rate
            profit_to_vault = last_trade_profit * self.vault_rate
            self.vault_balance += profit_to_vault
        
        # Risk capital is everything EXCEPT the vault money
        self.risk_capital = self.total_balance - self.vault_balance
        
        # Check if we graduated to a new phase
        self._update_growth_phase()

    def calculate_lot_size(self, stop_loss_pips: int, confidence_score: float, 
                           pip_value: float = 10.0, base_risk_pct: float = None) -> float:
        """
        Dynamic Lot Sizing: Snowball Aggressor Logic.
        
        Args:
            stop_loss_pips: Distance to SL.
            confidence_score: Titanium Density (0.0 - 1.0).
        """
        if stop_loss_pips <= 0: return 0.01
        
        # If base_risk is not provided, use the Phase Maximum
        if base_risk_pct is None:
            base_risk_pct = self.max_risk_pct

        # AGGRESSOR LOGIC:
        # If confidence > 90% (Titanium Floor), we go FULL Phase Risk.
        # If confidence is lower, we scale down quadratically.
        
        # Example in Phase 1 (Max 5%):
        # Conf 0.95 -> Risk 4.5% (Aggressive)
        # Conf 0.60 -> Risk 1.8% (Conservative)
        
        dynamic_risk_pct = base_risk_pct * (confidence_score ** 2)
        
        # Safety Clamp: Never exceed the phase limit
        dynamic_risk_pct = min(dynamic_risk_pct, self.max_risk_pct)
        
        risk_amount = self.risk_capital * dynamic_risk_pct
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        return round(max(0.01, lot_size), 2)

    def get_fund_status(self) -> dict:
        return {
            "phase": self.phase_name,
            "total_equity": self.total_balance,
            "risk_capital": self.risk_capital,
            "vault_balance": self.vault_balance,
            "active_vault_rate": self.vault_rate,
            "max_risk_allowed": self.max_risk_pct
        }

# Singleton initialized with $20 for Micro-Account simulation
risk_officer = MoneyManager(20.0)
