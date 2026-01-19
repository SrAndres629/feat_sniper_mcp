import sys
import os
import pandas as pd
import numpy as np
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexus_core.kinetic_engine import KineticEngine
from nexus_core.fundamental_engine.engine import FundamentalEngine
from nexus_core.fundamental_engine.calendar_client import CalendarClient

def test_kinetic_fix():
    print("\n🏎️ [TRIDENT] KINETIC FORMULA TEST")
    print("-----------------------------------")
    # Create a scenario where Volume is High but RVOL is Low (e.g. consistently high volume)
    # vs Volume High and RVOL High.
    
    # We just need to ensure the code RUNS and produces the metric 'prev_force'
    # And specifically, we want to verify it doesn't crash on the new RVOL math.
    
    dates = pd.date_range(start="2023-01-01", periods=50, freq="1H")
    df = pd.DataFrame({
        "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0
    }, index=dates)
    
    # Spike volume at end to verify calculation
    df.iloc[-2, df.columns.get_loc("volume")] = 5000.0 # T-1 Big Vol
    df.iloc[-2, df.columns.get_loc("close")] = 110.0  # T-1 Big Move
    
    engine = KineticEngine()
    metrics = engine.compute_kinetic_state(df)
    
    print(f"   ► Prev Force: {metrics.get('prev_force', 'N/A')}")
    print(f"   ► FEAT Force: {metrics.get('feat_force', 'N/A')}")
    
    if "prev_force" in metrics:
        print("   ✅ SUCCESS: Kinetic Engine patched and operational.")
    else:
        print("   ❌ FAIL: prev_force missing.")

def test_fundamental_engine():
    print("\n🌍 [TRIDENT] FUNDAMENTAL ENGINE TEST")
    print("-----------------------------------")
    
    # 1. Test Mock (Should Always Work)
    try:
        print("   ► Testing MOCK Provider...")
        engine_mock = FundamentalEngine(calendar_provider="mock")
        res_mock = engine_mock.check_event_proximity()
        print(f"     Status: {res_mock['macro_regime']}")
        print(f"     Events Found: {res_mock['next_event'].event_name if res_mock['next_event'] else 'None'}")
        print("   ✅ SUCCESS: Mock Provider verified.")
    except Exception as e:
        print(f"   ❌ FAIL: Mock Provider died. {e}")

    # 2. Test Real (ForexFactory) - Might fail if URL is bad, but we want to see it TRY.
    print("   ► Testing REAL Provider (ForexFactory)...")
    try:
        engine_real = FundamentalEngine(calendar_provider="forexfactory")
        # This calls the method that does Requests.
        # It might print warnings/errors from the class itself.
        res_real = engine_real.check_event_proximity()
        print(f"     Status: {res_real['macro_regime']}")
        next_ev = res_real.get("next_event")
        if next_ev:
             print(f"     Next Event: {next_ev.event_name} ({next_ev.impact})")
        else:
             print("     Next Event: None Found (Network or Parsing?)")
             
        print("   ⚠️ NOTE: Real Provider test completed (Result depends on Network/Supabase).")
    except Exception as e:
        print(f"   ❌ FAIL: Real Provider crashed. {e}")

if __name__ == "__main__":
    print("🔱 TRIDENT DIAGNOSTIC SUITE")
    test_kinetic_fix()
    test_fundamental_engine()
