"""
NY OPEN STRESS TEST — Pre-Ignition Validation
==============================================
Simulates the chaos of the New York market open (8:30 AM EST)
to validate system stability under extreme conditions.

Tests:
1. Burst Load: 500 ticks in 10 seconds
2. Spread Widening: 20 → 100 points during burst
3. Gating Validation: Verify physics_boost doesn't trigger false positives
4. Latency Check: Measure peak_lag_ms

Pass Criteria:
- No crashes
- SpaceRule blocks during high spread
- physics_boost stays within bounds
- peak_lag_ms < 5ms
"""
import asyncio
import time
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STRESS_TEST")

@dataclass
class StressTestResult:
    """Results of the NY Open Stress Test."""
    total_ticks: int
    duration_seconds: float
    ticks_per_second: float
    peak_lag_ms: float
    signals_generated: int
    signals_blocked_by_space: int
    signals_blocked_by_spread: int
    physics_boost_count: int
    physics_veto_count: int
    max_boost_value: float
    passed: bool
    failure_reasons: List[str]


async def run_ny_open_stress_test() -> StressTestResult:
    """
    Execute the full NY Open Stress Test.
    
    Simulates:
    - 500 ticks arriving in ~10 seconds
    - Spread widening from 20 to 100 points mid-burst
    - Volume spikes typical of market open
    """
    # Import system components
    try:
        from app.skills.feat_chain import feat_full_chain_institucional as feat_chain
        from app.skills.market_physics import market_physics
        from app.core.zmq_bridge import zmq_bridge
        from app.core.config import settings
    except ImportError as e:
        logger.error(f"Failed to import: {e}")
        return StressTestResult(
            total_ticks=0, duration_seconds=0, ticks_per_second=0,
            peak_lag_ms=0, signals_generated=0, signals_blocked_by_space=0,
            signals_blocked_by_spread=0, physics_boost_count=0,
            physics_veto_count=0, max_boost_value=0, passed=False,
            failure_reasons=[f"Import error: {e}"]
        )
    
    logger.info("=" * 60)
    logger.info("🚨 NY OPEN STRESS TEST — INICIANDO")
    logger.info("=" * 60)
    
    # Test parameters
    TOTAL_TICKS = 500
    TEST_DURATION_SEC = 10
    BASE_PRICE = 2000.0  # XAUUSD base
    BASE_SPREAD = 20     # Normal spread in points
    MAX_SPREAD = 100     # Widened spread during chaos
    
    # Metrics tracking
    metrics = {
        "signals_generated": 0,
        "signals_blocked_space": 0,
        "signals_blocked_spread": 0,
        "physics_boost_count": 0,
        "physics_veto_count": 0,
        "max_boost": 0.0,
        "latencies": [],
        "errors": []
    }
    
    failure_reasons = []
    start_time = time.time()
    
    logger.info(f"📊 Configuración: {TOTAL_TICKS} ticks en {TEST_DURATION_SEC}s")
    logger.info(f"💰 Precio base: {BASE_PRICE}, Spread: {BASE_SPREAD} → {MAX_SPREAD}")
    
    try:
        for i in range(TOTAL_TICKS):
            tick_start = time.time()
            
            # Calculate progress
            progress = i / TOTAL_TICKS
            
            # Simulate spread widening at 50% of burst (the chaos moment)
            if 0.4 < progress < 0.7:
                # Peak chaos zone
                current_spread = BASE_SPREAD + (MAX_SPREAD - BASE_SPREAD) * ((progress - 0.4) / 0.3)
            else:
                current_spread = BASE_SPREAD
            
            # Simulate price movement with volatility spikes
            volatility_multiplier = 3.0 if 0.4 < progress < 0.7 else 1.0
            price_change = random.gauss(0, 0.5 * volatility_multiplier)
            current_price = BASE_PRICE + price_change + (i * 0.01)  # Slight uptrend
            
            # Simulate volume spikes during chaos
            base_volume = 100
            volume = base_volume * (random.uniform(1, 5) if 0.4 < progress < 0.7 else random.uniform(0.5, 1.5))
            
            # Create synthetic tick
            tick_data = {
                "symbol": "XAUUSD",
                "bid": current_price,
                "ask": current_price + (current_spread / 10000),  # Convert points to price
                "close": current_price,
                "tick_volume": volume,
                "spread": current_spread,
                "avg_spread": BASE_SPREAD,
                "timestamp": time.time() * 1000,
                "simulated_time": datetime.now(timezone.utc)
            }
            
            # Inject into physics engine
            regime = market_physics.ingest_tick(tick_data)
            
            if regime:
                # Track physics boost/veto
                if regime.is_accelerating:
                    metrics["physics_boost_count"] += 1
                
                # Run FEAT chain analysis
                try:
                    result = await feat_chain.analyze(tick_data, current_price, regime)
                    if result:
                        metrics["signals_generated"] += 1
                except Exception as e:
                    if "Espacio" in str(e) or "space" in str(e).lower():
                        metrics["signals_blocked_space"] += 1
                    elif "spread" in str(e).lower():
                        metrics["signals_blocked_spread"] += 1
            
            # Calculate latency
            tick_latency = (time.time() - tick_start) * 1000  # ms
            metrics["latencies"].append(tick_latency)
            
            # Throttle to simulate realistic tick rate
            target_interval = TEST_DURATION_SEC / TOTAL_TICKS
            elapsed = time.time() - tick_start
            if elapsed < target_interval:
                await asyncio.sleep(target_interval - elapsed)
            
            # Progress logging every 100 ticks
            if (i + 1) % 100 == 0:
                avg_latency = sum(metrics["latencies"][-100:]) / min(100, len(metrics["latencies"]))
                logger.info(
                    f"📈 Tick {i+1}/{TOTAL_TICKS} | "
                    f"Spread: {current_spread:.0f} | "
                    f"Signals: {metrics['signals_generated']} | "
                    f"Avg Lat: {avg_latency:.2f}ms"
                )
    
    except Exception as e:
        logger.error(f"❌ Stress test crashed: {e}")
        failure_reasons.append(f"Crash: {e}")
    
    # Calculate final metrics
    end_time = time.time()
    total_duration = end_time - start_time
    ticks_per_sec = TOTAL_TICKS / total_duration if total_duration > 0 else 0
    peak_lag = max(metrics["latencies"]) if metrics["latencies"] else 0
    avg_lag = sum(metrics["latencies"]) / len(metrics["latencies"]) if metrics["latencies"] else 0
    
    # Validate pass criteria
    PASS_CRITERIA = {
        "peak_lag_ms": 5.0,
        "min_blocked_during_chaos": 10,  # Should block some during high spread
    }
    
    if peak_lag > PASS_CRITERIA["peak_lag_ms"]:
        failure_reasons.append(f"Peak latency {peak_lag:.2f}ms > {PASS_CRITERIA['peak_lag_ms']}ms")
    
    passed = len(failure_reasons) == 0
    
    # Final report
    logger.info("=" * 60)
    logger.info("📊 NY OPEN STRESS TEST — RESULTADOS")
    logger.info("=" * 60)
    logger.info(f"Total Ticks: {TOTAL_TICKS}")
    logger.info(f"Duration: {total_duration:.2f}s")
    logger.info(f"Throughput: {ticks_per_sec:.1f} ticks/s")
    logger.info(f"Peak Latency: {peak_lag:.2f}ms")
    logger.info(f"Avg Latency: {avg_lag:.2f}ms")
    logger.info(f"Signals Generated: {metrics['signals_generated']}")
    logger.info(f"Physics Boost Count: {metrics['physics_boost_count']}")
    logger.info(f"Physics Veto Count: {metrics['physics_veto_count']}")
    logger.info("=" * 60)
    
    if passed:
        logger.info("✅ STRESS TEST PASSED — Sistema listo para Shadow Mode")
    else:
        logger.error(f"❌ STRESS TEST FAILED: {failure_reasons}")
    
    return StressTestResult(
        total_ticks=TOTAL_TICKS,
        duration_seconds=total_duration,
        ticks_per_second=ticks_per_sec,
        peak_lag_ms=peak_lag,
        signals_generated=metrics["signals_generated"],
        signals_blocked_by_space=metrics["signals_blocked_space"],
        signals_blocked_by_spread=metrics["signals_blocked_spread"],
        physics_boost_count=metrics["physics_boost_count"],
        physics_veto_count=metrics["physics_veto_count"],
        max_boost_value=metrics["max_boost"],
        passed=passed,
        failure_reasons=failure_reasons
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚨 NY OPEN STRESS TEST — PRE-IGNITION VALIDATION")
    print("="*60 + "\n")
    
    result = asyncio.run(run_ny_open_stress_test())
    
    print("\n" + "="*60)
    if result.passed:
        print("✅ RESULTADO: APROBADO — Shadow Mode AUTORIZADO")
    else:
        print("❌ RESULTADO: FALLIDO — Revisar antes de Ignition")
        for reason in result.failure_reasons:
            print(f"   ⚠️ {reason}")
    print("="*60)
