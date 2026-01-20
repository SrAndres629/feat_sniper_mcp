"""
FEAT SNIPER: SNOWBALL GROWTH SIMULATOR (Monte Carlo)
====================================================
Simulates the growth of a $20 micro-account using the new Phase-Based MoneyManager.
Verifies 'Escape Velocity' from the Survival Zone (<$100).
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from nexus_core.money_management import MoneyManager

def simulate_growth_path(
    starting_balance: float = 20.0,
    win_rate: float = 0.72,
    reward_risk_ratio: float = 2.0,  # 2:1 RR
    n_trades: int = 100,
    titanium_prob: float = 0.20  # 20% of signals are high-confidence Titanium
):
    """
    Runs a single simulation of N trades.
    """
    mm = MoneyManager(starting_balance)
    balance_history = [starting_balance]
    vault_history = [0.0]
    phases_log = []
    
    for _ in range(n_trades):
        # 1. Determine Signal Quality (Titanium or Normal)
        is_titanium = np.random.random() < titanium_prob
        confidence = 0.95 if is_titanium else 0.65
        
        # 2. Determine Outcome (Win/Loss)
        is_win = np.random.random() < win_rate
        
        # 3. Calculate Lot Size & Risk
        # Assumption: standard trade risks 20 pips
        sl_pips = 20
        pip_value = 1.0 # Micro-lot value assumption
        
        # Calculate Risk Amount ($)
        # We need to reverse-engineer the lot size to get dollar risk
        lot_size = mm.calculate_lot_size(sl_pips, confidence, pip_value)
        risk_dollars = lot_size * sl_pips * pip_value
        
        # 4. Calculate PnL
        if is_win:
            pnl = risk_dollars * reward_risk_ratio
        else:
            pnl = -risk_dollars
            
        # 5. Update Manager
        mm.update_balance(pnl)
        
        status = mm.get_fund_status()
        balance_history.append(status['total_equity'])
        vault_history.append(status['vault_balance'])
        phases_log.append(status['phase'])
        
        if status['total_equity'] <= 0: # Ruin
            break
            
    return balance_history, vault_history, phases_log

def run_monte_carlo_simulation(n_sims: int = 1000):
    print(f"=== OPERATION SNOWBALL: MONTE CARLO SIMULATION ({n_sims} runs) ===")
    print("Parameters: Start=$20 | WinRate=72% | RR=2:1 | TitaniumFreq=20%\n")
    
    success_count = 0
    ruin_count = 0
    escape_velocities = [] # Trades needed to reach $100
    final_balances = []
    
    for i in range(n_sims):
        balances, _, _ = simulate_growth_path(n_trades=200)
        final_bal = balances[-1]
        final_balances.append(final_bal)
        
        if final_bal <= 0:
            ruin_count += 1
        elif final_bal >= 100:
            success_count += 1
            # Find trade # where we crossed $100
            for idx, bal in enumerate(balances):
                if bal >= 100:
                    escape_velocities.append(idx)
                    break
    
    avg_escape = np.mean(escape_velocities) if escape_velocities else 0
    ruin_prob = (ruin_count / n_sims) * 100
    median_final = np.median(final_balances)
    
    print("-" * 40)
    print(f"RESULTS:")
    print(f"💰 Median Final Balance (200 trades): ${median_final:.2f}")
    print(f"🚀 Probability of Escaping Survival Zone (>$100): {(success_count/n_sims)*100:.1f}%")
    print(f"☠️ Probability of Ruin (Bust): {ruin_prob:.1f}%")
    print(f"⏳ Avg Trades to Escape Phase 1 ($20->$100): {avg_escape:.1f} trades")
    print("-" * 40)
    
    if ruin_prob < 5.0:
        print("✅ VERDICT: SAFE AGGRESSION. Tuning Accepted.")
    else:
        print("⚠️ VERDICT: TOO RISKY. Reduce Phase 1 Risk Parameter.")

if __name__ == "__main__":
    run_monte_carlo_simulation()
