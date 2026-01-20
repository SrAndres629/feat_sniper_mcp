"""
FEAT SNIPER: STRATEGY ENGINE SIMULATION
=======================================
Verifies the Tactician's logic:
1. Twin Trading (Split positions on Titanium events)
2. Dynamic Target Promotion (Scalp -> Swing)
3. Bias Enforcement (No Sell Swings)
"""

import sys
sys.path.insert(0, '.')
from nexus_core.strategy_engine import StrategyEngine, StrategyMode, TradeLeg
from nexus_core.money_management import MoneyManager

def test_strategy_logic():
    print("=== OPERATION DYNAMIC EXECUTION: LOGIC AUDIT ===")
    
    # Setup
    mm = MoneyManager(100.0) # $100 account
    engine = StrategyEngine(mm)
    
    # CASE 1: TITANIUM BUY (High Prob Scalp + Swing Allowed)
    print("\n[CASE 1] Titanium BUY Signal (P_Scalp=0.9, P_Swing=0.2)")
    legs = engine.analyze_strategic_intent(
        market_price=2000.0,
        neural_probs={'scalp': 0.9, 'day': 0.3, 'swing': 0.2},
        macro_context={'direction': 'BUY'},
        titanium_level=True
    )
    
    print(f"-> Generated {len(legs)} trade legs:")
    for i, leg in enumerate(legs):
        print(f"   Leg {i+1}: {leg.direction} | Mode: {leg.strategy_type.value} | Intent: {leg.intent}")
        
    if len(legs) == 2 and legs[1].strategy_type == StrategyMode.SWING:
        print("✅ SUCCESS: Twin Trading activated correctly (Scalp + Swing Runner).")
    else:
        print("❌ FAILURE: Twin Trading logic failed.")

    # CASE 2: TITANIUM SELL (High Prob Scalp + Swing Blocked)
    print("\n[CASE 2] Titanium SELL Signal (P_Scalp=0.9)")
    legs = engine.analyze_strategic_intent(
        market_price=2000.0,
        neural_probs={'scalp': 0.9, 'day': 0.3, 'swing': 0.2},
        macro_context={'direction': 'SELL'}, # SELLING
        titanium_level=True
    )
    
    print(f"-> Generated {len(legs)} trade legs:")
    has_swing = any(l.strategy_type == StrategyMode.SWING for l in legs)
    if not has_swing and len(legs) == 2:
        print("✅ SUCCESS: Sell Swing blocked. Downgraded to Day Runner.")
    elif has_swing:
        print("❌ FAILURE: Strategy Engine allowed a SELL SWING (Violation of Prime Directive).")
        
    # CASE 3: DYNAMIC PROMOTION
    print("\n[CASE 3] Dynamic Promotion (Scalp -> Swing)")
    active_legs = [
        TradeLeg("BUY", 0.01, 1990, 2010, StrategyMode.SCALP, "CASH_FLOW")
    ]
    
    # New check: Swing prob spikes to 0.85
    promoted = engine.optimize_targets(
        active_trades=active_legs,
        fresh_probs={'scalp': 0.6, 'day': 0.7, 'swing': 0.85}
    )
    
    if promoted and active_legs[0].strategy_type == StrategyMode.SWING:
        print("✅ SUCCESS: Scalp promoted to Swing Runner.")
    else:
        print("❌ FAILURE: Promotion logic failed.")

if __name__ == "__main__":
    test_strategy_logic()
