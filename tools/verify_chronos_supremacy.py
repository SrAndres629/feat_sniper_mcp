import sys
import os
import datetime
import pytz
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from app.ml.feat_processor.time import TimeFeatureProcessor
from nexus_core.chronos_engine.phaser import GoldCyclePhaser
from nexus_core.risk_engine.guards import KillZoneGuard

def run_test():
    print("🛸 CHRONOS SUPREMACY: GOLD MICROSTRUCTURE DIAGNOSTIC")
    print("=====================================================")
    
    master_proc = TimeFeatureProcessor()
    phaser = GoldCyclePhaser() 
    guard = KillZoneGuard()
    
    # 1. Generate Synthetic Data
    prices_rw = pd.Series(np.cumsum(np.random.normal(0, 1, 200)) + 1000)
    vols = pd.Series(np.random.randint(100, 1000, 200)) # Low Volume
    vols_high = pd.Series(np.random.randint(2000, 5000, 200)) # High Volume
    opens = prices_rw.shift(1).fillna(1000)
    
    # --- SIMULATION 1: LONDON RAID (02:15 Bolivia) ---
    # User Doctrine: "02:00-03:00 is RAID. No Breakouts."
    t_london = datetime.datetime(2026, 1, 20, 6, 15, tzinfo=datetime.timezone.utc) # 02:15 UTC-4
    
    print("\n[SCENARIO 1] Time: 02:15 Bolivia (London Raid)")
    payload = master_proc.process(t_london, prices_rw, vols, opens)
    state_lon = phaser.get_current_state(t_london)
    
    print(f"   ► Phase:       {payload['debug_phase']}")
    print(f"   ► Intent:      {state_lon.intent.name}")
    print(f"   ► Guard Info:  {state_lon.action_guard}")
    
    # Test Guard: Attempt a Breakout
    guard_res = guard.check_trade_allowed("BREAKOUT", state_lon, 500, 500)
    print(f"   🛑 GUARD CHECK (Breakout): {guard_res['allowed']} -> {guard_res['reason']}")
    
    if not guard_res['allowed'] and "LONDON_RAID" in str(state_lon.micro_phase):
         print("   ✅ SUCCESS: Guard Blocked Breakout in Trap Zone.")
    else:
         print("   ❌ FAIL: Guard Failed.")

    # --- SIMULATION 2: NY CONFIRMATION (09:15 Bolivia) ---
    # User Doctrine: "09:00-09:30 Confirm with Volume"
    t_ny = datetime.datetime(2026, 1, 20, 13, 15, tzinfo=datetime.timezone.utc) # 09:15 UTC-4
    
    print("\n[SCENARIO 2] Time: 09:15 Bolivia (NY Confirmation)")
    payload_ny = master_proc.process(t_ny, prices_rw, vols, opens) # Low Vol
    state_ny = phaser.get_current_state(t_ny)
    
    # Test Guard with LOW Volume
    guard_res_low = guard.check_trade_allowed("BREAKOUT", state_ny, 500, 600)
    print(f"   ► GUARD CHECK (Low Vol):  {guard_res_low['allowed']} -> {guard_res_low['reason']}")
    
    # Test Guard with HIGH Volume
    guard_res_high = guard.check_trade_allowed("BREAKOUT", state_ny, 2000, 600)
    print(f"   ► GUARD CHECK (High Vol): {guard_res_high['allowed']} -> {guard_res_high['reason']}")
    
    if not guard_res_low['allowed'] and guard_res_high['allowed']:
         print("   ✅ SUCCESS: Guard Enforced Volume Confirmation.")
    else:
         print("   ❌ FAIL: Volume Logic Failed.")

    print("=====================================================")

if __name__ == "__main__":
    run_test()
