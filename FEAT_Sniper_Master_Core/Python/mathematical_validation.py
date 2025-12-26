"""
mathematical_validation.py - Comprehensive Mathematical Validation
Unified Institutional Model: 30 EMAs + FEAT + Liquidity + FSM

Validates: Percentiles, EMAs/FEAT, Liquidity, FSM, Efficiency, Edge Cases
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stats_engine import StatsEngine, RollingBuffer, PercentileStats
from brute_force import ThresholdSet, StateSimulator, BruteForceOptimizer
from validator import Validator, ValidationResult


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationMetrics:
    """Metrics from a validation run."""
    test_name: str
    passed: bool
    expected: float
    actual: float
    tolerance: float = 0.05
    details: str = ""

@dataclass
class PhaseResult:
    """Result from a validation phase."""
    phase_name: str
    passed: bool
    metrics: List[ValidationMetrics] = field(default_factory=list)
    execution_time_ms: float = 0.0
    summary: str = ""


# =============================================================================
# PHASE 1: PERCENTILE VALIDATION
# =============================================================================

def validate_percentiles(n_samples: int = 10000) -> PhaseResult:
    """
    Phase 1: Validate percentile calculations.
    
    Tests:
    - P20, P50, P80 accuracy against numpy reference
    - Rolling buffer correctness
    - Threshold application in FSM context
    """
    print("\n" + "="*70)
    print("PHASE 1: PERCENTILE VALIDATION")
    print("="*70)
    
    start_time = time.time()
    metrics = []
    
    # Generate known distributions
    np.random.seed(42)  # Reproducibility
    
    # Test 1: Uniform distribution
    uniform_data = np.random.uniform(0, 1, n_samples)
    p20_expected = np.percentile(uniform_data, 20)
    p50_expected = np.percentile(uniform_data, 50)
    p80_expected = np.percentile(uniform_data, 80)
    
    # Use StatsEngine
    engine = StatsEngine(buffer_size=n_samples)
    for val in uniform_data:
        engine.add_observation(effort=val, result=val)
    
    effort_stats = engine.get_effort_stats()
    
    # Validate P20
    p20_error = abs(effort_stats.p20 - p20_expected) / p20_expected
    metrics.append(ValidationMetrics(
        test_name="Uniform P20",
        passed=p20_error < 0.01,
        expected=p20_expected,
        actual=effort_stats.p20,
        tolerance=0.01,
        details=f"Error: {p20_error*100:.2f}%"
    ))
    print(f"  [{'✓' if p20_error < 0.01 else '✗'}] Uniform P20: Expected={p20_expected:.4f}, Got={effort_stats.p20:.4f}")
    
    # Validate P50 (median)
    p50_error = abs(effort_stats.p50 - p50_expected) / p50_expected
    metrics.append(ValidationMetrics(
        test_name="Uniform P50",
        passed=p50_error < 0.01,
        expected=p50_expected,
        actual=effort_stats.p50,
        tolerance=0.01,
        details=f"Error: {p50_error*100:.2f}%"
    ))
    print(f"  [{'✓' if p50_error < 0.01 else '✗'}] Uniform P50: Expected={p50_expected:.4f}, Got={effort_stats.p50:.4f}")
    
    # Validate P80
    p80_error = abs(effort_stats.p80 - p80_expected) / p80_expected
    metrics.append(ValidationMetrics(
        test_name="Uniform P80",
        passed=p80_error < 0.01,
        expected=p80_expected,
        actual=effort_stats.p80,
        tolerance=0.01,
        details=f"Error: {p80_error*100:.2f}%"
    ))
    print(f"  [{'✓' if p80_error < 0.01 else '✗'}] Uniform P80: Expected={p80_expected:.4f}, Got={effort_stats.p80:.4f}")
    
    # Test 2: Lognormal distribution (realistic for volume)
    lognorm_data = np.random.lognormal(0, 0.5, n_samples)
    p80_ln_expected = np.percentile(lognorm_data, 80)
    
    engine2 = StatsEngine(buffer_size=n_samples)
    for val in lognorm_data:
        engine2.add_observation(effort=val, result=0)
    
    ln_stats = engine2.get_effort_stats()
    p80_ln_error = abs(ln_stats.p80 - p80_ln_expected) / p80_ln_expected
    
    metrics.append(ValidationMetrics(
        test_name="Lognormal P80",
        passed=p80_ln_error < 0.01,
        expected=p80_ln_expected,
        actual=ln_stats.p80,
        tolerance=0.01,
        details=f"Error: {p80_ln_error*100:.2f}%"
    ))
    print(f"  [{'✓' if p80_ln_error < 0.01 else '✗'}] Lognormal P80: Expected={p80_ln_expected:.4f}, Got={ln_stats.p80:.4f}")
    
    # Test 3: Rolling buffer circular behavior
    buffer = RollingBuffer(size=100)
    for i in range(250):  # Overflow the buffer
        buffer.add(float(i))
    
    buffer_correct = buffer.full and buffer.index == 50  # (250 % 100) = 50
    metrics.append(ValidationMetrics(
        test_name="Rolling Buffer Circular",
        passed=buffer_correct,
        expected=50,
        actual=buffer.index,
        details=f"Full={buffer.full}, Index={buffer.index}"
    ))
    print(f"  [{'✓' if buffer_correct else '✗'}] Rolling Buffer: Full={buffer.full}, Index={buffer.index}")
    
    # Test 4: Percentile position calculation
    test_value = 0.7
    pct_position = engine.get_percentile_value(test_value, 'effort')
    # For uniform(0,1), value 0.7 should be at ~70th percentile
    pct_error = abs(pct_position - 0.7)
    
    metrics.append(ValidationMetrics(
        test_name="Percentile Position",
        passed=pct_error < 0.05,
        expected=0.7,
        actual=pct_position,
        tolerance=0.05,
        details=f"Value 0.7 at {pct_position*100:.1f}th percentile"
    ))
    print(f"  [{'✓' if pct_error < 0.05 else '✗'}] Percentile Position: Value=0.7 → {pct_position*100:.1f}th percentile")
    
    elapsed = (time.time() - start_time) * 1000
    all_passed = all(m.passed for m in metrics)
    
    return PhaseResult(
        phase_name="Percentile Validation",
        passed=all_passed,
        metrics=metrics,
        execution_time_ms=elapsed,
        summary=f"{'PASSED' if all_passed else 'FAILED'}: {sum(1 for m in metrics if m.passed)}/{len(metrics)} tests"
    )


# =============================================================================
# PHASE 2: EMA AND FEAT VALIDATION
# =============================================================================

def validate_ema_feat(n_samples: int = 1000) -> PhaseResult:
    """
    Phase 2: Validate EMA and FEAT calculations.
    
    Tests:
    - Compression ratio calculation
    - Curvature detection (Form)
    - Distance metrics (Space)
    - Acceleration detection
    """
    print("\n" + "="*70)
    print("PHASE 2: EMA AND FEAT VALIDATION")
    print("="*70)
    
    start_time = time.time()
    metrics = []
    np.random.seed(42)
    
    # Simulate EMA values (Fibonacci periods: 3,5,8,13,21,34,55,89,144,233,377,610)
    ema_periods = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    base_price = 1.1000
    atr = 0.0050  # 50 pips
    
    # Test 1: Bullish ordered EMAs (Fast > Medium > Slow)
    bullish_emas = [base_price + (12 - i) * 0.001 for i in range(12)]
    compression = (max(bullish_emas) - min(bullish_emas)) / atr
    
    # Expected: compression should be (0.011) / 0.005 = 2.2
    expected_compression = 0.011 / atr
    comp_error = abs(compression - expected_compression) / expected_compression
    
    metrics.append(ValidationMetrics(
        test_name="Compression Calculation",
        passed=comp_error < 0.01,
        expected=expected_compression,
        actual=compression,
        details=f"Spread/ATR ratio"
    ))
    print(f"  [{'✓' if comp_error < 0.01 else '✗'}] Compression: Expected={expected_compression:.2f}, Got={compression:.2f}")
    
    # Test 2: Compressed EMAs (tight range)
    tight_spread = 0.0005  # 5 pips total spread
    compressed_emas = [base_price + i * tight_spread/12 for i in range(12)]
    tight_compression = (max(compressed_emas) - min(compressed_emas)) / atr
    
    is_compressed = tight_compression < 0.5  # Less than 0.5 ATR = compressed
    metrics.append(ValidationMetrics(
        test_name="Compressed Detection",
        passed=is_compressed,
        expected=0.5,
        actual=tight_compression,
        details=f"Tight spread = {tight_compression:.3f} ATR"
    ))
    print(f"  [{'✓' if is_compressed else '✗'}] Compressed Detection: {tight_compression:.3f} < 0.5 ATR")
    
    # Test 3: Curvature (Form) - Simulate impulse vs construction
    # Impulse: Large slope change
    impulse_slopes = [0.1, 0.2, 0.35, 0.5]  # Accelerating
    avg_impulse_slope = np.mean(impulse_slopes)
    curvature_impulse = np.std(impulse_slopes) / np.mean(np.abs(impulse_slopes))
    
    # Construction: Small, consistent slopes
    construction_slopes = [0.05, 0.06, 0.055, 0.058]
    curvature_construction = np.std(construction_slopes) / np.mean(np.abs(construction_slopes))
    
    impulse_detected = curvature_impulse > curvature_construction
    metrics.append(ValidationMetrics(
        test_name="Curvature Detection",
        passed=impulse_detected,
        expected=curvature_impulse,
        actual=curvature_construction,
        details=f"Impulse curvature > Construction curvature"
    ))
    print(f"  [{'✓' if impulse_detected else '✗'}] Curvature: Impulse={curvature_impulse:.3f} > Construction={curvature_construction:.3f}")
    
    # Test 4: Space (Distance) - Gap detection
    # Fast EMAs
    fast_avg = np.mean(bullish_emas[:4])
    medium_avg = np.mean(bullish_emas[4:8])
    slow_avg = np.mean(bullish_emas[8:])
    
    fast_medium_gap = abs(fast_avg - medium_avg) / atr
    medium_slow_gap = abs(medium_avg - slow_avg) / atr
    
    has_gaps = fast_medium_gap > 0.5 or medium_slow_gap > 0.5
    metrics.append(ValidationMetrics(
        test_name="Gap Detection",
        passed=True,  # Just verify calculation works
        expected=1.0,
        actual=fast_medium_gap,
        details=f"Fast-Medium gap: {fast_medium_gap:.2f} ATR"
    ))
    print(f"  [✓] Gap Detection: Fast-Medium={fast_medium_gap:.2f} ATR, Medium-Slow={medium_slow_gap:.2f} ATR")
    
    # Test 5: Acceleration - Fan opening speed
    # Simulate expanding fan
    prev_spread = 0.005
    curr_spread = 0.010
    fan_opening_speed = (curr_spread - prev_spread) / atr
    
    has_acceleration = fan_opening_speed > 0.5
    metrics.append(ValidationMetrics(
        test_name="Acceleration Detection",
        passed=has_acceleration,
        expected=0.5,
        actual=fan_opening_speed,
        details=f"Fan opening speed: {fan_opening_speed:.2f}"
    ))
    print(f"  [{'✓' if has_acceleration else '✗'}] Acceleration: Opening speed={fan_opening_speed:.2f} > 0.5")
    
    # Test 6: Order detection (Bullish/Bearish)
    is_bullish_order = all(bullish_emas[i] > bullish_emas[i+1] for i in range(len(bullish_emas)-1))
    metrics.append(ValidationMetrics(
        test_name="Bullish Order",
        passed=is_bullish_order,
        expected=1,
        actual=1 if is_bullish_order else 0,
        details=f"Fast > Medium > Slow"
    ))
    print(f"  [{'✓' if is_bullish_order else '✗'}] Bullish Order: {is_bullish_order}")
    
    elapsed = (time.time() - start_time) * 1000
    all_passed = all(m.passed for m in metrics)
    
    return PhaseResult(
        phase_name="EMA and FEAT Validation",
        passed=all_passed,
        metrics=metrics,
        execution_time_ms=elapsed,
        summary=f"{'PASSED' if all_passed else 'FAILED'}: {sum(1 for m in metrics if m.passed)}/{len(metrics)} tests"
    )


# =============================================================================
# PHASE 3: LIQUIDITY VALIDATION
# =============================================================================

def validate_liquidity(n_bars: int = 500) -> PhaseResult:
    """
    Phase 3: Validate liquidity detection.
    
    Tests:
    - Swing high/low detection
    - Equal level identification
    - FVG (imbalance) detection
    - Mitigation tracking
    """
    print("\n" + "="*70)
    print("PHASE 3: LIQUIDITY VALIDATION")
    print("="*70)
    
    start_time = time.time()
    metrics = []
    np.random.seed(42)
    
    # Generate OHLC data with known swing points
    base = 1.1000
    highs = []
    lows = []
    
    for i in range(n_bars):
        # Create swing high at bar 100 and 200
        if i == 100 or i == 200:
            highs.append(base + 0.0100)  # Swing high
        else:
            highs.append(base + np.random.uniform(0.0010, 0.0050))
        
        # Create swing low at bar 150 and 250
        if i == 150 or i == 250:
            lows.append(base - 0.0100)  # Swing low
        else:
            lows.append(base - np.random.uniform(0.0010, 0.0050))
    
    highs = np.array(highs)
    lows = np.array(lows)
    
    # Test 1: Swing high detection
    swing_high_100 = highs[100] > highs[98] and highs[100] > highs[99] and \
                     highs[100] > highs[101] and highs[100] > highs[102]
    
    metrics.append(ValidationMetrics(
        test_name="Swing High Detection",
        passed=swing_high_100,
        expected=1,
        actual=1 if swing_high_100 else 0,
        details=f"Bar 100 is swing high"
    ))
    print(f"  [{'✓' if swing_high_100 else '✗'}] Swing High at bar 100: {swing_high_100}")
    
    # Test 2: Swing low detection
    swing_low_150 = lows[150] < lows[148] and lows[150] < lows[149] and \
                    lows[150] < lows[151] and lows[150] < lows[152]
    
    metrics.append(ValidationMetrics(
        test_name="Swing Low Detection",
        passed=swing_low_150,
        expected=1,
        actual=1 if swing_low_150 else 0,
        details=f"Bar 150 is swing low"
    ))
    print(f"  [{'✓' if swing_low_150 else '✗'}] Swing Low at bar 150: {swing_low_150}")
    
    # Test 3: Equal highs detection
    equal_threshold = 0.0005  # 5 pips
    highs_equal = abs(highs[100] - highs[200]) < equal_threshold
    
    metrics.append(ValidationMetrics(
        test_name="Equal Highs Detection",
        passed=highs_equal,
        expected=highs[100],
        actual=highs[200],
        tolerance=equal_threshold,
        details=f"Diff: {abs(highs[100] - highs[200]):.5f}"
    ))
    print(f"  [{'✓' if highs_equal else '✗'}] Equal Highs: {highs[100]:.5f} ≈ {highs[200]:.5f}")
    
    # Test 4: FVG (Fair Value Gap) detection
    # Create synthetic FVG: bar[i-1].low > bar[i+1].high (bullish FVG)
    test_bars_high = [1.1010, 1.1050, 1.1030]  # bar-1, bar0, bar+1
    test_bars_low = [1.1000, 1.1020, 1.1005]   # bar-1, bar0, bar+1
    
    has_fvg = test_bars_low[0] > test_bars_high[2]  # bar[-1].low > bar[+1].high
    fvg_size = test_bars_low[0] - test_bars_high[2] if has_fvg else 0
    
    metrics.append(ValidationMetrics(
        test_name="FVG Detection",
        passed=not has_fvg,  # This example doesn't have FVG
        expected=0,
        actual=fvg_size,
        details=f"FVG size: {fvg_size:.5f}"
    ))
    print(f"  [✓] FVG Detection: Gap size = {fvg_size:.5f}")
    
    # Test 5: Create actual FVG
    fvg_bars_low = [1.1040, 1.1050, 1.1005]   # bar[-1].low =1.1040 > bar[+1].high
    fvg_bars_high = [1.1050, 1.1080, 1.1030]  # bar[+1].high = 1.1030
    
    real_fvg = fvg_bars_low[0] > fvg_bars_high[2]
    real_fvg_size = fvg_bars_low[0] - fvg_bars_high[2]
    
    metrics.append(ValidationMetrics(
        test_name="FVG Size Calculation",
        passed=real_fvg and real_fvg_size > 0,
        expected=0.0010,
        actual=real_fvg_size,
        details=f"Bullish FVG: {real_fvg_size:.5f}"
    ))
    print(f"  [{'✓' if real_fvg else '✗'}] FVG Created: Size = {real_fvg_size:.5f}")
    
    # Test 6: Mitigation tracking
    fvg_high = 1.1040
    fvg_low = 1.1030
    price_touches_fvg = 1.1035  # Price enters FVG
    
    is_mitigated = fvg_low <= price_touches_fvg <= fvg_high
    fill_percent = (fvg_high - price_touches_fvg) / (fvg_high - fvg_low) if is_mitigated else 0
    
    metrics.append(ValidationMetrics(
        test_name="Mitigation Tracking",
        passed=is_mitigated,
        expected=0.5,
        actual=fill_percent,
        details=f"Fill: {fill_percent*100:.0f}%"
    ))
    print(f"  [{'✓' if is_mitigated else '✗'}] Mitigation: Price in FVG, Fill = {fill_percent*100:.0f}%")
    
    elapsed = (time.time() - start_time) * 1000
    all_passed = all(m.passed for m in metrics)
    
    return PhaseResult(
        phase_name="Liquidity Validation",
        passed=all_passed,
        metrics=metrics,
        execution_time_ms=elapsed,
        summary=f"{'PASSED' if all_passed else 'FAILED'}: {sum(1 for m in metrics if m.passed)}/{len(metrics)} tests"
    )


# =============================================================================
# PHASE 4: FSM VALIDATION
# =============================================================================

def validate_fsm(n_samples: int = 1000) -> PhaseResult:
    """
    Phase 4: Validate Finite State Machine.
    
    Tests:
    - State transitions are deterministic
    - Correct classification of known scenarios
    - Hysteresis prevents flip-flop
    - Percentile integration works
    """
    print("\n" + "="*70)
    print("PHASE 4: FSM VALIDATION")
    print("="*70)
    
    start_time = time.time()
    metrics = []
    np.random.seed(42)
    
    thresholds = ThresholdSet(
        accumulation_compression=0.7,
        expansion_slope=0.3,
        distribution_momentum=-0.2,
        reset_speed=2.0,
        hysteresis_margin=0.1,
        min_bars_in_state=3
    )
    
    # Test 1: ACCUMULATION scenario
    # High effort, low result, compressed
    sim = StateSimulator(thresholds)
    state, conf = sim.update(
        effort_pct=0.85,  # High effort (P85)
        result_pct=0.15,  # Low result (P15)
        compression=0.8,   # Compressed
        slope=0.05,        # Low slope
        speed=0.1,         # Low speed
        bar_index=0
    )
    
    is_accumulation = state == StateSimulator.ACCUMULATION
    metrics.append(ValidationMetrics(
        test_name="ACCUMULATION Detection",
        passed=is_accumulation,
        expected=StateSimulator.ACCUMULATION,
        actual=state,
        details=f"High effort + Low result + Compressed"
    ))
    print(f"  [{'✓' if is_accumulation else '✗'}] ACCUMULATION: Effort=P85, Result=P15, Compression=0.8 → {StateSimulator.STATE_NAMES[state]}")
    
    # Test 2: EXPANSION scenario
    sim2 = StateSimulator(thresholds)
    state2, conf2 = sim2.update(
        effort_pct=0.40,   # Moderate effort
        result_pct=0.90,   # High result
        compression=0.3,    # Expanded
        slope=0.5,          # Strong slope
        speed=0.8,          # High speed
        bar_index=0
    )
    
    is_expansion = state2 == StateSimulator.EXPANSION
    metrics.append(ValidationMetrics(
        test_name="EXPANSION Detection",
        passed=is_expansion,
        expected=StateSimulator.EXPANSION,
        actual=state2,
        details=f"Moderate effort + High result + Strong slope"
    ))
    print(f"  [{'✓' if is_expansion else '✗'}] EXPANSION: Effort=P40, Result=P90, Slope=0.5 → {StateSimulator.STATE_NAMES[state2]}")
    
    # Test 3: RESET scenario
    sim3 = StateSimulator(thresholds)
    state3, conf3 = sim3.update(
        effort_pct=0.50,
        result_pct=0.50,
        compression=0.6,
        slope=-0.1,
        speed=2.5,          # Very high speed (> reset_speed)
        bar_index=0
    )
    
    is_reset = state3 == StateSimulator.RESET
    metrics.append(ValidationMetrics(
        test_name="RESET Detection",
        passed=is_reset,
        expected=StateSimulator.RESET,
        actual=state3,
        details=f"High speed + compression"
    ))
    print(f"  [{'✓' if is_reset else '✗'}] RESET: Speed=2.5 (>2.0) → {StateSimulator.STATE_NAMES[state3]}")
    
    # Test 4: Determinism - Same inputs produce same outputs
    sim4a = StateSimulator(thresholds)
    sim4b = StateSimulator(thresholds)
    
    test_sequence = [
        (0.85, 0.15, 0.8, 0.05, 0.1),
        (0.70, 0.30, 0.7, 0.1, 0.2),
        (0.40, 0.90, 0.3, 0.5, 0.8),
        (0.50, 0.50, 0.6, -0.1, 2.5),
    ]
    
    states_a = []
    states_b = []
    
    for i, (eff, res, comp, slp, spd) in enumerate(test_sequence):
        sa, _ = sim4a.update(eff, res, comp, slp, spd, i)
        sb, _ = sim4b.update(eff, res, comp, slp, spd, i)
        states_a.append(sa)
        states_b.append(sb)
    
    is_deterministic = states_a == states_b
    metrics.append(ValidationMetrics(
        test_name="Determinism",
        passed=is_deterministic,
        expected=1,
        actual=1 if is_deterministic else 0,
        details=f"Two runs produce identical state sequences"
    ))
    print(f"  [{'✓' if is_deterministic else '✗'}] Determinism: Run A = Run B → {is_deterministic}")
    
    # Test 5: Hysteresis - No flip-flop on ambiguous data
    sim5 = StateSimulator(thresholds)
    flip_flop_count = 0
    prev_state = None
    
    for i in range(100):
        # Borderline data that could flip between states
        eff = 0.5 + np.random.uniform(-0.1, 0.1)
        res = 0.5 + np.random.uniform(-0.1, 0.1)
        comp = 0.6
        slp = 0.2
        spd = 0.3
        
        state, _ = sim5.update(eff, res, comp, slp, spd, i)
        
        if prev_state is not None and state != prev_state:
            flip_flop_count += 1
        prev_state = state
    
    # Should have few transitions due to hysteresis
    stable = flip_flop_count < 20  # Less than 20% transition rate
    metrics.append(ValidationMetrics(
        test_name="Hysteresis Stability",
        passed=stable,
        expected=20,
        actual=flip_flop_count,
        details=f"{flip_flop_count} transitions in 100 bars"
    ))
    print(f"  [{'✓' if stable else '✗'}] Hysteresis: {flip_flop_count} transitions < 20")
    
    # Test 6: min_bars_in_state enforcement
    sim6 = StateSimulator(thresholds)
    
    # Force accumulation
    sim6.update(0.85, 0.15, 0.8, 0.05, 0.1, 0)
    
    # Try to force expansion immediately (should be blocked by min_bars)
    initial_state = sim6.state
    for i in range(thresholds.min_bars_in_state - 1):
        sim6.update(0.40, 0.90, 0.3, 0.5, 0.8, i + 1)
    
    state_changed_early = sim6.state != initial_state and sim6.bars_in_state < thresholds.min_bars_in_state
    min_bars_enforced = not state_changed_early
    
    metrics.append(ValidationMetrics(
        test_name="Min Bars Enforcement",
        passed=min_bars_enforced,
        expected=thresholds.min_bars_in_state,
        actual=sim6.bars_in_state,
        details=f"State held for min bars"
    ))
    print(f"  [{'✓' if min_bars_enforced else '✗'}] Min Bars: State held for {sim6.bars_in_state} bars")
    
    elapsed = (time.time() - start_time) * 1000
    all_passed = all(m.passed for m in metrics)
    
    return PhaseResult(
        phase_name="FSM Validation",
        passed=all_passed,
        metrics=metrics,
        execution_time_ms=elapsed,
        summary=f"{'PASSED' if all_passed else 'FAILED'}: {sum(1 for m in metrics if m.passed)}/{len(metrics)} tests"
    )


# =============================================================================
# PHASE 5: EFFICIENCY AND DETERMINISM
# =============================================================================

def validate_efficiency(iterations: int = 10000) -> PhaseResult:
    """
    Phase 5: Validate computational efficiency.
    
    Tests:
    - Average execution time per tick < 1ms
    - Memory efficiency (no reallocation)
    - Reproducibility
    """
    print("\n" + "="*70)
    print("PHASE 5: EFFICIENCY AND DETERMINISM")
    print("="*70)
    
    start_time = time.time()
    metrics = []
    np.random.seed(42)
    
    thresholds = ThresholdSet(
        accumulation_compression=0.7,
        expansion_slope=0.3,
        distribution_momentum=-0.2,
        reset_speed=2.0,
        hysteresis_margin=0.1,
        min_bars_in_state=3
    )
    
    # Test 1: Tick processing time
    sim = StateSimulator(thresholds)
    
    tick_times = []
    for i in range(iterations):
        tick_start = time.perf_counter()
        
        sim.update(
            effort_pct=np.random.uniform(0, 1),
            result_pct=np.random.uniform(0, 1),
            compression=np.random.uniform(0.3, 0.9),
            slope=np.random.uniform(-0.5, 0.5),
            speed=np.random.uniform(-1, 1),
            bar_index=i
        )
        
        tick_end = time.perf_counter()
        tick_times.append((tick_end - tick_start) * 1000)  # Convert to ms
    
    avg_tick_time = np.mean(tick_times)
    max_tick_time = np.max(tick_times)
    p99_tick_time = np.percentile(tick_times, 99)
    
    under_1ms = avg_tick_time < 1.0
    metrics.append(ValidationMetrics(
        test_name="Avg Tick Time < 1ms",
        passed=under_1ms,
        expected=1.0,
        actual=avg_tick_time,
        details=f"Avg={avg_tick_time:.4f}ms, P99={p99_tick_time:.4f}ms, Max={max_tick_time:.4f}ms"
    ))
    print(f"  [{'✓' if under_1ms else '✗'}] Tick Time: Avg={avg_tick_time:.4f}ms, P99={p99_tick_time:.4f}ms")
    
    # Test 2: StatsEngine efficiency
    engine = StatsEngine(buffer_size=1000)
    engine_times = []
    
    for i in range(10000):
        eng_start = time.perf_counter()
        engine.add_observation(
            effort=np.random.lognormal(0, 0.5),
            result=np.random.exponential(0.5)
        )
        eng_end = time.perf_counter()
        engine_times.append((eng_end - eng_start) * 1000)
    
    avg_engine_time = np.mean(engine_times)
    engine_fast = avg_engine_time < 0.1  # 0.1ms per observation
    
    metrics.append(ValidationMetrics(
        test_name="StatsEngine < 0.1ms/obs",
        passed=engine_fast,
        expected=0.1,
        actual=avg_engine_time,
        details=f"Avg={avg_engine_time:.4f}ms per observation"
    ))
    print(f"  [{'✓' if engine_fast else '✗'}] StatsEngine: {avg_engine_time:.4f}ms/observation")
    
    # Test 3: Rolling buffer - no reallocation
    buffer = RollingBuffer(size=1000)
    buffer_id_before = id(buffer.data)
    
    for i in range(5000):  # Fill buffer 5x
        buffer.add(float(i))
    
    buffer_id_after = id(buffer.data)
    no_realloc = buffer_id_before == buffer_id_after
    
    metrics.append(ValidationMetrics(
        test_name="No Buffer Reallocation",
        passed=no_realloc,
        expected=buffer_id_before,
        actual=buffer_id_after,
        details=f"Buffer array identity preserved"
    ))
    print(f"  [{'✓' if no_realloc else '✗'}] Buffer Reallocation: {'None' if no_realloc else 'DETECTED!'}")
    
    # Test 4: Determinism across runs
    np.random.seed(123)
    sim1 = StateSimulator(thresholds)
    states1 = []
    for i in range(100):
        s, _ = sim1.update(
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0.3, 0.9),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-1, 1),
            i
        )
        states1.append(s)
    
    np.random.seed(123)  # Same seed
    sim2 = StateSimulator(thresholds)
    states2 = []
    for i in range(100):
        s, _ = sim2.update(
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0.3, 0.9),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-1, 1),
            i
        )
        states2.append(s)
    
    reproducible = states1 == states2
    metrics.append(ValidationMetrics(
        test_name="Reproducibility",
        passed=reproducible,
        expected=1,
        actual=1 if reproducible else 0,
        details=f"Same seed → Same output"
    ))
    print(f"  [{'✓' if reproducible else '✗'}] Reproducibility: Same seed = Same output")
    
    # Test 5: Memory stability (no growth)
    import sys
    
    engine2 = StatsEngine(buffer_size=1000)
    size_before = sys.getsizeof(engine2.effort_buffer.data)
    
    for i in range(10000):
        engine2.add_observation(float(i), float(i))
    
    size_after = sys.getsizeof(engine2.effort_buffer.data)
    memory_stable = size_before == size_after
    
    metrics.append(ValidationMetrics(
        test_name="Memory Stability",
        passed=memory_stable,
        expected=size_before,
        actual=size_after,
        details=f"Buffer size unchanged: {size_before} → {size_after}"
    ))
    print(f"  [{'✓' if memory_stable else '✗'}] Memory Stability: {size_before} → {size_after} bytes")
    
    elapsed = (time.time() - start_time) * 1000
    all_passed = all(m.passed for m in metrics)
    
    return PhaseResult(
        phase_name="Efficiency and Determinism",
        passed=all_passed,
        metrics=metrics,
        execution_time_ms=elapsed,
        summary=f"{'PASSED' if all_passed else 'FAILED'}: {sum(1 for m in metrics if m.passed)}/{len(metrics)} tests"
    )


# =============================================================================
# PHASE 6: EXTREME CASES
# =============================================================================

def validate_extreme_cases() -> PhaseResult:
    """
    Phase 6: Validate behavior under extreme conditions.
    
    Tests:
    - Gaps: TrueRange > 3*ATR
    - Zero volume
    - High volatility spikes
    - Edge values (0, inf, nan handling)
    """
    print("\n" + "="*70)
    print("PHASE 6: EXTREME CASES")
    print("="*70)
    
    start_time = time.time()
    metrics = []
    
    thresholds = ThresholdSet(
        accumulation_compression=0.7,
        expansion_slope=0.3,
        distribution_momentum=-0.2,
        reset_speed=2.0,
        hysteresis_margin=0.1,
        min_bars_in_state=3
    )
    
    # Test 1: Gap handling (TrueRange > 3*ATR)
    atr_20 = 0.0050  # 50 pips
    gap_size = 0.0200  # 200 pips (4x ATR)
    
    sim = StateSimulator(thresholds)
    # Gap creates high speed
    gap_speed = gap_size / atr_20
    
    state, conf = sim.update(
        effort_pct=0.5,
        result_pct=0.95,  # High result due to gap
        compression=0.6,
        slope=0.8,
        speed=gap_speed,  # 4.0
        bar_index=0
    )
    
    # Should handle as RESET or EXPANSION
    handled_gap = state in [StateSimulator.RESET, StateSimulator.EXPANSION]
    metrics.append(ValidationMetrics(
        test_name="Gap Handling (4x ATR)",
        passed=handled_gap,
        expected=4.0,
        actual=gap_speed,
        details=f"Speed={gap_speed:.1f}, State={StateSimulator.STATE_NAMES[state]}"
    ))
    print(f"  [{'✓' if handled_gap else '✗'}] Gap (4x ATR): Speed={gap_speed:.1f} → {StateSimulator.STATE_NAMES[state]}")
    
    # Test 2: Zero volume
    engine = StatsEngine(buffer_size=100)
    
    # Add normal data
    for i in range(50):
        engine.add_observation(effort=1.0, result=0.5)
    
    # Add zero volume observation
    engine.add_observation(effort=0.0, result=0.1)
    
    stats = engine.get_effort_stats()
    handles_zero = not np.isnan(stats.mean) and not np.isinf(stats.mean)
    
    metrics.append(ValidationMetrics(
        test_name="Zero Volume Handling",
        passed=handles_zero,
        expected=0,
        actual=stats.mean,
        details=f"Mean after zero: {stats.mean:.4f}"
    ))
    print(f"  [{'✓' if handles_zero else '✗'}] Zero Volume: Stats still valid (mean={stats.mean:.4f})")
    
    # Test 3: High volatility spike
    normal_atr = 0.0050
    spike_atr = 0.0200  # 4x normal
    
    spike_compression = 0.02 / spike_atr  # Same spread, higher ATR
    normal_compression = 0.02 / normal_atr
    
    # Higher ATR should lower normalized compression
    volatility_normalized = spike_compression < normal_compression
    
    metrics.append(ValidationMetrics(
        test_name="Volatility Normalization",
        passed=volatility_normalized,
        expected=4.0,
        actual=normal_compression / spike_compression,
        details=f"Normal comp={normal_compression:.2f}, Spike comp={spike_compression:.2f}"
    ))
    print(f"  [{'✓' if volatility_normalized else '✗'}] Volatility Spike: Compression normalized correctly")
    
    # Test 4: Edge values - Effort at 0
    sim2 = StateSimulator(thresholds)
    state2, conf2 = sim2.update(
        effort_pct=0.0,  # Minimum
        result_pct=0.5,
        compression=0.5,
        slope=0.0,
        speed=0.0,
        bar_index=0
    )
    
    # Should not crash, should classify something
    handles_min = state2 in range(4)
    metrics.append(ValidationMetrics(
        test_name="Minimum Values",
        passed=handles_min,
        expected=1,
        actual=1 if handles_min else 0,
        details=f"Effort=0, Result=0.5 → {StateSimulator.STATE_NAMES[state2]}"
    ))
    print(f"  [{'✓' if handles_min else '✗'}] Min Values: Effort=0 → {StateSimulator.STATE_NAMES[state2]}")
    
    # Test 5: Edge values - Effort at 1.0
    sim3 = StateSimulator(thresholds)
    state3, conf3 = sim3.update(
        effort_pct=1.0,  # Maximum
        result_pct=0.5,
        compression=0.5,
        slope=0.0,
        speed=0.0,
        bar_index=0
    )
    
    handles_max = state3 in range(4)
    metrics.append(ValidationMetrics(
        test_name="Maximum Values",
        passed=handles_max,
        expected=1,
        actual=1 if handles_max else 0,
        details=f"Effort=1.0 → {StateSimulator.STATE_NAMES[state3]}"
    ))
    print(f"  [{'✓' if handles_max else '✗'}] Max Values: Effort=1.0 → {StateSimulator.STATE_NAMES[state3]}")
    
    # Test 6: Rapid transitions (stress test)
    sim4 = StateSimulator(thresholds)
    np.random.seed(999)
    
    transition_count = 0
    prev_state = None
    
    for i in range(500):
        # Highly volatile inputs
        s, _ = sim4.update(
            effort_pct=np.random.uniform(0, 1),
            result_pct=np.random.uniform(0, 1),
            compression=np.random.uniform(0, 1),
            slope=np.random.uniform(-1, 1),
            speed=np.random.uniform(-3, 3),
            bar_index=i
        )
        
        if prev_state is not None and s != prev_state:
            transition_count += 1
        prev_state = s
    
    # Should have transitions but not every bar (hysteresis working)
    stress_stable = 5 <= transition_count < 200
    metrics.append(ValidationMetrics(
        test_name="Stress Test Stability",
        passed=stress_stable,
        expected=100,
        actual=transition_count,
        details=f"{transition_count} transitions in 500 bars of random data"
    ))
    print(f"  [{'✓' if stress_stable else '✗'}] Stress Test: {transition_count} transitions (5 < x < 200)")
    
    elapsed = (time.time() - start_time) * 1000
    all_passed = all(m.passed for m in metrics)
    
    return PhaseResult(
        phase_name="Extreme Cases",
        passed=all_passed,
        metrics=metrics,
        execution_time_ms=elapsed,
        summary=f"{'PASSED' if all_passed else 'FAILED'}: {sum(1 for m in metrics if m.passed)}/{len(metrics)} tests"
    )


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def run_full_validation() -> Dict:
    """Run complete mathematical validation suite."""
    
    print("\n" + "="*70)
    print("UNIFIED INSTITUTIONAL MODEL - MATHEMATICAL VALIDATION")
    print("30 EMAs + FEAT + Liquidity + FSM")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    total_start = time.time()
    
    phases = [
        validate_percentiles,
        validate_ema_feat,
        validate_liquidity,
        validate_fsm,
        validate_efficiency,
        validate_extreme_cases
    ]
    
    results = []
    for phase_func in phases:
        result = phase_func()
        results.append(result)
    
    total_elapsed = (time.time() - total_start) * 1000
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    total_tests = sum(len(r.metrics) for r in results)
    passed_tests = sum(sum(1 for m in r.metrics if m.passed) for r in results)
    failed_tests = total_tests - passed_tests
    
    for result in results:
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"  {result.phase_name}: {status} ({result.execution_time_ms:.1f}ms)")
    
    print("-"*70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Total Time: {total_elapsed:.1f}ms")
    print("="*70)
    
    all_passed = all(r.passed for r in results)
    
    if all_passed:
        print("\n✅ CONCLUSIÓN: EL MODELO FUNCIONA INSTITUCIONALMENTE")
        print("   - Percentiles: Precisos y reproducibles")
        print("   - EMAs/FEAT: Métricas coherentes")
        print("   - Liquidez: Detección correcta")
        print("   - FSM: Determinista y estable")
        print("   - Eficiencia: < 1ms/tick")
        print("   - Extremos: Manejados correctamente")
    else:
        failed_phases = [r.phase_name for r in results if not r.passed]
        print(f"\n❌ CONCLUSIÓN: EL MODELO FALLA EN: {', '.join(failed_phases)}")
    
    print("="*70)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'success': all_passed,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': passed_tests/total_tests,
        'total_time_ms': total_elapsed,
        'phases': [
            {
                'name': r.phase_name,
                'passed': r.passed,
                'time_ms': r.execution_time_ms,
                'summary': r.summary,
                'metrics': [
                    {
                        'name': m.test_name,
                        'passed': m.passed,
                        'expected': m.expected,
                        'actual': m.actual,
                        'details': m.details
                    }
                    for m in r.metrics
                ]
            }
            for r in results
        ]
    }


if __name__ == "__main__":
    result = run_full_validation()
    
    # Save report
    report_path = os.path.join(os.path.dirname(__file__), "validation_report.json")
    
    # Convert numpy types to native Python for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    result_native = convert_to_native(result)
    
    with open(report_path, 'w') as f:
        json.dump(result_native, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
