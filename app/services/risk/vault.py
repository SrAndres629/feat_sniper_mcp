import os
import json
import logging
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger("Risk.Vault")

class TheVault:
    """Capital Protection System (Level 66). Locks 50% profits once capital is doubled."""
    VAULT_STATE_FILE = "data/vault_state.json"
    COMPOUNDING_MULTIPLIER = 2.0
    VAULT_PERCENTAGE = 0.50
    
    def __init__(self, initial_capital: float = 30.0):
        self.initial_capital = initial_capital
        self.vault_balance = 0.0
        self.trading_capital = initial_capital
        self.total_vault_transfers = 0
        self.last_trigger_equity = initial_capital
        self._load_state()
    
    def _load_state(self):
        try:
            if os.path.exists(self.VAULT_STATE_FILE):
                with open(self.VAULT_STATE_FILE, "r") as f:
                    d = json.load(f)
                    self.initial_capital = d.get("initial_capital", self.initial_capital)
                    self.vault_balance = d.get("vault_balance", 0.0)
                    self.trading_capital = d.get("trading_capital", self.initial_capital)
                    self.total_vault_transfers = d.get("total_vault_transfers", 0)
                    self.last_trigger_equity = d.get("last_trigger_equity", self.initial_capital)
        except Exception as e: logger.warning(f"Vault load failure: {e}")
    
    def _save_state(self):
        os.makedirs(os.path.dirname(self.VAULT_STATE_FILE) or ".", exist_ok=True)
        data = {"initial_capital": self.initial_capital, "vault_balance": self.vault_balance, "trading_capital": self.trading_capital, "total_vault_transfers": self.total_vault_transfers, "last_trigger_equity": self.last_trigger_equity, "last_updated": datetime.now(timezone.utc).isoformat()}
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.VAULT_STATE_FILE), suffix=".tmp")
        try:
            with os.fdopen(fd, 'w') as f: json.dump(data, f, indent=2)
            os.replace(tmp, self.VAULT_STATE_FILE)
        except Exception as e:
            if os.path.exists(tmp): os.remove(tmp)
            logger.error(f"Vault atomic save failed: {e}")
    
    def check_vault_trigger(self, current_equity: float) -> Optional[Dict[str, Any]]:
        target = self.last_trigger_equity * self.COMPOUNDING_MULTIPLIER
        if current_equity >= target:
            profit = current_equity - self.last_trigger_equity
            v_amt = profit * self.VAULT_PERCENTAGE
            r_amt = profit - v_amt
            self.vault_balance += v_amt
            self.trading_capital = self.last_trigger_equity + r_amt
            self.last_trigger_equity = self.trading_capital
            self.total_vault_transfers += 1
            self._save_state()
            return {"vault_amount": v_amt, "total_vaulted": self.vault_balance, "new_trading_capital": self.trading_capital}
        return None
