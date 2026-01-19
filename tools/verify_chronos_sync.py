import sys
import os
import datetime
import pytz
sys.path.append(os.getcwd())

from nexus_core.fundamental_engine.risk_modulator import RiskModulator
from nexus_core.chronos_engine.tracker import FractalTimeTracker, IPDAGlobalState, KillZone
from app.ml.feat_processor.time_flow import TemporalTensorEngine

def run_test():
    print("⏳ CHRONOS ENGINE: VERIFICATION PROTOCOL")
    print("=" * 50)
    
    tracker = FractalTimeTracker()
    tensor_engine = TemporalTensorEngine()
    risk_engine = RiskModulator()
    
    # Test Cases: (UTC Time) -> we need to approximate what UTC matches specific NY times
    # Assuming Winter time (UTC-5)
    
    # Case 1: NY OPEN (09:30 AM NY) -> 14:30 UTC
    verify_time(tracker, tensor_engine, risk_engine, datetime.datetime(2026, 1, 20, 14, 30, tzinfo=datetime.timezone.utc), "NY_OPEN_0930")

    # Case 2: ASIA LATE (22:00 PM NY) -> 03:00 UTC (Next Day)
    verify_time(tracker, tensor_engine, risk_engine, datetime.datetime(2026, 1, 20, 3, 0, tzinfo=datetime.timezone.utc), "ASIA_2200") # Actually 22:00 prev day

    # Case 3: LONDON OPEN (03:00 AM NY) -> 08:00 UTC
    verify_time(tracker, tensor_engine, risk_engine, datetime.datetime(2026, 1, 20, 8, 0, tzinfo=datetime.timezone.utc), "LONDON_0300")
    
    print("\n" + "=" * 50)
    print("✅ TEST COMPLETE")

def verify_time(tracker, tensor_engine, risk_engine, dt, label):
    state = tracker.get_market_state(dt)
    
    # Layer 2 Calculations
    cyclic = tensor_engine.calculate_cyclic_time(state.timestamp_ny)
    intensity = tensor_engine.calculate_killzone_intensity(state.timestamp_ny)
    fractal_tensor = tensor_engine.encode_fractal_phase(state.fractal_hour)
    
    # Layer 3 Calculations
    base_risk = 1.0
    final_risk = risk_engine.apply_chronos_factor(base_risk, state.ipda_phase, state.is_kill_zone_active)
    
    print(f"\n[{label}] Input UTC: {dt.time()} -> NY Time: {state.timestamp_ny.strftime('%H:%M')}")
    print(f"   ► IPDA Phase:   {state.ipda_phase.name}")
    print(f"   ► 4H Fractal:   {state.fractal_hour} (Hour of Block) -> Tensor: {fractal_tensor}")
    print(f"   ► Kill Zone:    {state.kill_zone.name} (Active: {state.is_kill_zone_active})")
    print(f"   ► Quality Score: {state.liquidity_quality_score}")
    print(f"   ► Tensor Math:   Sin={cyclic['time_sin']:.3f}, Cos={cyclic['time_cos']:.3f}")
    print(f"   ► KZ Intensity:  {intensity:.4f} (Gaussian)")
    print(f"   ► RISK FACTORS:  Base={base_risk} -> Final={final_risk:.2f} (Shield Active)")

    
    # Basic Assertions
    if label == "NY_OPEN_0930":
        if state.ipda_phase != IPDAGlobalState.EXPANSION: print("   ❌ FAILED: Expected EXPANSION")
        if final_risk < 1.0: print("   ❌ FAILED: Risk should be boosted (>1.0)")
        if intensity < 0.9: print(f"   ❌ FAILED: Intensity too low ({intensity})")

    if label == "ASIA_2200":
        if final_risk > 0.15: print("   ❌ FAILED: Risk should be near zero")

if __name__ == "__main__":
    run_test()
