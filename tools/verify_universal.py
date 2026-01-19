import sys
import os
import datetime
import pytz

sys.path.append(os.getcwd())

from nexus_core.chronos_engine.phaser import GoldCyclePhaser
from nexus_core.risk_engine.guards import LiquidityRiskAuditor

from app.ml.feat_processor.chronos_tensor import ChronosTensorFactory

def run_test():
    print("🌍 ADAPTIVE CHRONOS: PHASE 2 DIAGNOSTIC")
    print("========================================")
    
    phaser = GoldCyclePhaser()
    auditor = LiquidityRiskAuditor()
    tensor_factory = ChronosTensorFactory()
    
    # Scenarios (Bolivia Time)
    # 1. ASIA (21:00): Range Bound, Low Vol
    t_asia = datetime.datetime(2026, 1, 21, 1, 0, tzinfo=datetime.timezone.utc)
    
    # 2. BLACKOUT (18:30): No Trade
    t_black = datetime.datetime(2026, 1, 20, 22, 30, tzinfo=datetime.timezone.utc)
    
    # 3. NY CONFIRMATION (09:15): High Vol, Expansion
    t_ny = datetime.datetime(2026, 1, 20, 13, 15, tzinfo=datetime.timezone.utc)
    
    scenarios = [
        ("ASIA (21:00)", t_asia, "RANGE_BOUND", 0.5), # Risk 0.5
        ("BLACKOUT (18:30)", t_black, "NO_TRADE", 0.0),
        ("NY CONFIRM (09:15)", t_ny, "TREND_FOLLOWING", 1.0)
    ]
    
    for name, t, expected_strat, expected_risk in scenarios:
        print(f"\n[{name}]")
        state = phaser.get_current_state(t)
        audit = auditor.audit_risk(state, 1000, 1000)
        tensors = tensor_factory.process(t)
        
        print(f"   ► Strategy: {state.profile.strategy_mode.value}")
        print(f"   ► Risk Coeff: {audit['risk_coefficient']}")
        print(f"   ► Exp Vol:    {tensors['expected_volatility'][0]:.2f}")
        print(f"   ► Intent:     {state.intent.name}")
        
        # Validations
        if state.profile.strategy_mode.value == expected_strat:
            print("   ✅ SUCCESS: Strategy Mode Match.")
        else:
            print(f"   ❌ FAIL: Strategy Mismatch (Got {state.profile.strategy_mode.value})")
            
        if audit['risk_coefficient'] == expected_risk:
             print(f"   ✅ SUCCESS: Risk Coefficient Match ({expected_risk}).")
        else:
             print(f"   ❌ FAIL: Risk Mismatch ({audit['risk_coefficient']})")
             
        if "BLACKOUT" in name:
            if tensors['expected_volatility'][0] == 0.0:
                print("   ✅ SUCCESS: Volatility Zeroed.")
            else:
                print("   ❌ FAIL: Blackout Volatility not 0.0")

    print("========================================")

if __name__ == "__main__":
    run_test()
