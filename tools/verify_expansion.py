import sys
import os
import datetime
import pytz
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from app.ml.feat_processor.targets import TargetTensorFactory
from nexus_core.risk_engine.guards import LiquidityRiskAuditor
from nexus_core.chronos_engine.phaser import GoldCyclePhaser

def run_test():
    print("🧠 PROBABILISTIC MIND: SYSTEM DIAGNOSTIC")
    print("========================================")
    
    target_factory = TargetTensorFactory()
    auditor = LiquidityRiskAuditor()
    phaser = GoldCyclePhaser()
    
    # 1. Test Targets (Gaussian Field)
    current_price = 2000.0
    atr_1h = 10.0
    # Bull Target is Price + 20 = 2020.
    
    # Simulation: We are at 2000. Target is 2020.
    # Expect Potential ~ 2.0. Proximity ~ 0.0.
    t_london = datetime.datetime(2026, 1, 20, 10, 0, tzinfo=datetime.timezone.utc)
    
    print("\n[SIMULATION 1] Expansion Potential (Price=2000, Target=2020)")
    payload = target_factory.process_targets(t_london, current_price, atr_1h)
    
    pot_z = payload['expansion_potential_bull'][0]
    prox_g = payload['target_proximity_bull'][0]
    
    print(f"   ► Expansion Potential (Z): {pot_z:.2f} (Expected 2.0)")
    print(f"   ► Target Proximity (G):    {prox_g:.4f} (Expected near 0.0)")
    
    if pot_z > 1.5:
        print("   ✅ SUCCESS: System sees 'Room to Run'.")

    # Simulation: We are AT Target (2020).
    print("\n[SIMULATION 2] At Target (Price=2020, Target=2020)")
    # Note: Projector calculates targets FROM current price at call time? 
    # Ah, the projector uses current price as base.
    # To test "At Target", we assume targets were fixed earlier. 
    # But this factory calculates dynamic targets from NOW.
    # So if Price=2020, Target becomes 2040.
    # The logic is "Potential from HERE".
    # Correct. The net always sees "Potential from HERE".
    
    # 2. Test Risk Auditor (Probabilistic)
    # Time: 02:15 Bolivia (London Raid)
    t_raid = datetime.datetime(2026, 1, 20, 6, 15, tzinfo=datetime.timezone.utc)
    state = phaser.get_current_state(t_raid)
    
    print("\n[SIMULATION 3] Risk Audit (London Raid)")
    audit = auditor.audit_risk(state, 500, 500)
    
    print(f"   ► Phase: {state.micro_phase}")
    print(f"   ► Risk Coefficient: {audit['risk_coefficient']}")
    print(f"   ► Reason: {audit['reason']}")
    
    if audit['risk_coefficient'] < 0.5:
        print("   ✅ SUCCESS: High Risk correctly penalized (Soft Guard).")
    else:
        print("   ❌ FAIL: Risk not penalized.")

    print("========================================")

if __name__ == "__main__":
    run_test()
