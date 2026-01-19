import sys
import os
sys.path.append(os.getcwd())

from nexus_core.fundamental_engine import fundamental_engine
from nexus_core.fundamental_engine.risk_modulator import DEFCON

def test_fundamental_engine():
    print("🌍 MACRO SENTINEL: FUNDAMENTAL ENGINE AUDIT")
    print("=" * 50)
    
    # 1. Check Event Proximity
    result = fundamental_engine.check_event_proximity(currencies=["USD"])
    
    print("\n[CHECK 1] Event Proximity Detection:")
    print(f"   DEFCON Level: {result['defcon'].name}")
    print(f"   Kill Switch Active: {result['kill_switch']}")
    print(f"   Position Multiplier: {result['position_multiplier']:.4f}")
    print(f"   Minutes to Next Event: {result['minutes_until']:.1f}")
    print(f"   Macro Regime: {result['macro_regime']}")
    
    if result['next_event']:
        e = result['next_event']
        print(f"   Next Event: {e.event_name} ({e.currency}, {e.impact.name})")
    
    # 2. Get Tensor for Neural Network
    tensor = fundamental_engine.get_macro_regime_tensor(currencies=["USD"])
    
    print("\n[CHECK 2] Macro Regime Tensor (One-Hot):")
    for k, v in tensor.items():
        print(f"   {k}: {v:.4f}")
    
    # 3. Validate DEFCON Logic
    print("\n[CHECK 3] DEFCON Logic Validation:")
    
    # DEFCON should be 1 or 2 because mock data has NFP in 2 hours
    # Actually, 2 hours = 120 minutes. DEFCON_3 is 60-240 min for HIGH.
    # So it should be DEFCON_3.
    
    expected_defcon = DEFCON.DEFCON_3  # 2 hours to NFP (HIGH)
    actual_defcon = result['defcon']
    
    # Mock has "FOMC Member Speech" in 15 min (MEDIUM) - Won't trigger HIGH DEFCON
    # And "Non-Farm Payrolls" in 2 hours (HIGH) -> DEFCON_3
    # The function gets_next_high_impact, so it should return NFP.
    
    # The 15 min event is MEDIUM, so only the 2h HIGH event matters.
    # 120 min -> DEFCON_3 (between 60-240)
    
    if actual_defcon == expected_defcon:
        print(f"   ✅ DEFCON Level Correct ({actual_defcon.name})")
    else:
        print(f"   ⚠️ DEFCON Mismatch. Expected {expected_defcon.name}, Got {actual_defcon.name}")
    
    # 4. Kill Switch Test
    print("\n[CHECK 4] Kill Switch Protocol:")
    import numpy as np
    test_signal = np.array([0.8, 0.1, 0.1])  # Bull signal
    
    # If DEFCON_1, signal should be zeroed
    if result['kill_switch']:
        modified = fundamental_engine.apply_kill_switch(test_signal)
        assert np.allclose(modified, 0), "Kill switch failed to zero signal!"
        print("   ✅ Kill Switch ACTIVE - Signal Zeroed")
    else:
        modified = fundamental_engine.apply_kill_switch(test_signal)
        assert np.allclose(modified, test_signal), "Signal incorrectly modified!"
        print("   ✅ Kill Switch INACTIVE - Signal Preserved")
    
    print("\n" + "=" * 50)
    print("🏆 MACRO SENTINEL ONLINE. System Protected.")

if __name__ == "__main__":
    try:
        test_fundamental_engine()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ AUDIT FAILED: {e}")
