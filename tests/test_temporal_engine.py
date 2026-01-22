"""
[MODULE 05 - VERIFICATION]
Temporal Engine Zero-Debt Test Suite.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from nexus_core.temporal_engine.engine import temporal_engine
from nexus_core.temporal_engine.encoders import (
    sinusoidal_encode, killzone_intensity, get_session_name, encode_hour
)

def test_sinusoidal_continuity():
    """Test that 23:59 → 00:00 is smooth (no discontinuity)."""
    print("\n--- Test 1: Sinusoidal Continuity ---")
    
    # Hour 23.99 and Hour 0.01 should be close
    sin_23, cos_23 = encode_hour(23.99)
    sin_00, cos_00 = encode_hour(0.01)
    
    # Calculate Euclidean distance in 2D space
    distance = np.sqrt((sin_23 - sin_00)**2 + (cos_23 - cos_00)**2)
    print(f"23:59 encoding: ({sin_23:.4f}, {cos_23:.4f})")
    print(f"00:01 encoding: ({sin_00:.4f}, {cos_00:.4f})")
    print(f"Euclidean distance: {distance:.6f}")
    
    if distance < 0.1:
        print("✅ PASSED: Midnight continuity preserved.")
        return True
    else:
        print("❌ FAILED: Discontinuity at midnight.")
        return False

def test_killzone_ny_peak():
    """Test that NY Open (9:30 AM EST = 14:30 UTC) has high intensity."""
    print("\n--- Test 2: NY Killzone Peak ---")
    
    # NY Open peak around 14:00-14:30 UTC
    ny_intensity = killzone_intensity(14, 30)
    print(f"NY Open (14:30 UTC) intensity: {ny_intensity:.4f}")
    
    if ny_intensity > 0.8:
        print("✅ PASSED: NY Open has peak intensity.")
        return True
    else:
        print("❌ FAILED: NY Open intensity too low.")
        return False

def test_killzone_dead_zone():
    """Test that 3:00 AM EST (8:00 UTC during Asia) has low intensity."""
    print("\n--- Test 3: Dead Zone Intensity ---")
    
    # Asia session
    asia_intensity = killzone_intensity(3, 0)  # 3:00 UTC
    print(f"Asia Dead Zone (03:00 UTC) intensity: {asia_intensity:.4f}")
    
    if asia_intensity < 0.35:
        print("✅ PASSED: Asia dead zone has low intensity.")
        return True
    else:
        print("❌ FAILED: Asia intensity too high.")
        return False

def test_session_identification():
    """Test session name identification from UTC hour."""
    print("\n--- Test 4: Session Identification ---")
    
    test_cases = [
        (8, "LONDON"),
        (14, "NY_OPEN"),
        (16, "LONDON_CLOSE"),
        (3, "ASIA"),
        (19, "NY_LATE")
    ]
    
    all_passed = True
    for hour, expected in test_cases:
        actual = get_session_name(hour)
        status = "✅" if actual == expected else "❌"
        print(f"  {hour}:00 UTC → {actual} (expected: {expected}) {status}")
        if actual != expected:
            all_passed = False
    
    if all_passed:
        print("✅ PASSED: All sessions correctly identified.")
        return True
    else:
        print("❌ FAILED: Some sessions misidentified.")
        return False

def test_temporal_tensor():
    """Test full temporal tensor computation."""
    print("\n--- Test 5: Temporal Tensor ---")
    
    # Create synthetic data with datetime index
    n = 100
    times = pd.date_range("2024-01-15 08:00", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [100] * n,
        "high": [101] * n,
        "low": [99] * n,
        "close": np.linspace(100, 110, n),
        "volume": [10000] * n
    }, index=times)
    
    # Compute tensor
    df = temporal_engine.compute_temporal_tensor(df)
    
    # Check columns exist
    required_cols = ["temporal_sin", "temporal_cos", "killzone_intensity", "session_phase"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if not missing:
        print(f"All temporal columns present: {required_cols}")
        print(f"Killzone Intensity range: {df['killzone_intensity'].min():.2f} - {df['killzone_intensity'].max():.2f}")
        print("✅ PASSED: Temporal tensor computed correctly.")
        return True
    else:
        print(f"❌ FAILED: Missing columns: {missing}")
        return False


def test_weekly_cycle():
    """Test weekly cycle encoding."""
    print("\n--- Test 6: Weekly Cycle ---")
    from nexus_core.temporal_engine.encoders import encode_day_of_week, encode_weekly_phase, get_weekly_feat_phase
    
    # Test Friday-Monday continuity
    fri_sin, fri_cos = encode_day_of_week(4)  # Friday
    mon_sin, mon_cos = encode_day_of_week(0)  # Monday
    
    # Calculate distance in circular space
    distance = np.sqrt((fri_sin - mon_sin)**2 + (fri_cos - mon_cos)**2)
    print(f"Friday encoding: ({fri_sin:.4f}, {fri_cos:.4f})")
    print(f"Monday encoding: ({mon_sin:.4f}, {mon_cos:.4f})")
    print(f"Friday-Monday distance: {distance:.4f}")
    
    # Test weekly phase
    mon_phase = encode_weekly_phase(0, 9)  # Monday 9:00
    fri_phase = encode_weekly_phase(4, 15)  # Friday 15:00
    print(f"Monday 9:00 weekly phase: {mon_phase:.4f}")
    print(f"Friday 15:00 weekly phase: {fri_phase:.4f}")
    
    # Test FEAT phase names
    phases = [get_weekly_feat_phase(d) for d in range(5)]
    print(f"Weekly FEAT phases: {phases}")
    
    if distance < 2.0 and fri_phase > mon_phase:
        print("✅ PASSED: Weekly cycle correctly encoded.")
        return True
    else:
        print("❌ FAILED: Weekly cycle encoding issue.")
        return False


def run_all_tests():

    print("=" * 60)
    print("MODULE 05: TEMPORAL DIMENSION ENGINE VERIFICATION")
    print("=" * 60)
    
    results = []
    results.append(test_sinusoidal_continuity())
    results.append(test_killzone_ny_peak())
    results.append(test_killzone_dead_zone())
    results.append(test_session_identification())
    results.append(test_temporal_tensor())
    results.append(test_weekly_cycle())
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT: {passed}/{total} PASSED")
    print("=" * 60)
    
    if passed == total:
        print("🎯 MÓDULO 05 SELLADO. Consciencia Temporal + Weekly Cycle Activo.")
    else:
        print("⚠️ Requiere revisión adicional.")

if __name__ == "__main__":
    run_all_tests()
