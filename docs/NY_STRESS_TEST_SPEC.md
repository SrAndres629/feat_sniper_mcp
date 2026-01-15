# NY Open Stress Test - Technical Specification

> **Author**: Antigravity (Commander)  
> **Target**: Jules (Constructor)  
> **Date**: 2026-01-15

---

## 1. Objective

Simulate the **9:30 AM ET market open** tick burst to validate FEAT Sniper's latency invariants under extreme load.

## 2. Tick Burst Parameters

| Parameter               | Value | Rationale                    |
| ----------------------- | ----- | ---------------------------- |
| `ticks_per_second`      | 500   | Peak NY Open rate for EURUSD |
| `burst_duration_s`      | 60    | Standard stress window       |
| `volatility_multiplier` | 3.0   | Simulates gap/spike event    |
| `spread_expansion`      | 2.5x  | Realistic liquidity crunch   |

### Tick Generation Formula
```python
price_delta = np.random.normal(0, base_volatility * volatility_multiplier)
new_price = last_price + price_delta
spread = base_spread * spread_expansion * (1 + np.random.uniform(0, 0.5))
```

## 3. Failure Thresholds

| Metric             | PASS | WARN   | FAIL  |
| ------------------ | ---- | ------ | ----- |
| `peak_lag_ms`      | < 5  | 5-8    | > 8   |
| `avg_lag_ms`       | < 2  | 2-4    | > 4   |
| `dropped_ticks`    | 0    | 1-5    | > 5   |
| `memory_growth_mb` | < 50 | 50-100 | > 100 |

## 4. Pipeline Invariants to Assert

```python
# Each tick must complete full FEAT pipeline
assert tick.form_score is not None      # Form calculated
assert tick.fvg_detected is not None    # Space analyzed  
assert tick.acceleration >= -1.0        # Physics bounded
assert tick.hurst_valid == True         # Time calibrated
assert tick.signal_latency_ms < 5       # End-to-end OK
```

## 5. Module Structure

```
app/tests/stress_tester.py
├── TickBurstGenerator     # NumPy-based tick simulation
├── PipelineProfiler       # Latency measurement per stage
├── InvariantValidator     # FEAT constraint checker
└── ReportGenerator        # JSON output for CLI analysis
```

## 6. Integration Points

| Component      | Interface                 | Notes             |
| -------------- | ------------------------- | ----------------- |
| ZMQ Bridge     | `zmq_bridge.publish()`    | Inject ticks here |
| FEAT Chain     | `feat_chain.process()`    | Full pipeline     |
| CircuitBreaker | Must NOT trigger          | Use mock drawdown |
| Logger         | `logs/stress_test_*.json` | Results output    |

## 7. CLI Execution Command

```bash
python -m app.tests.stress_tester \
    --tps 500 \
    --duration 60 \
    --volatility 3.0 \
    --output logs/stress_test_$(date +%Y%m%d).json
```

## 8. Expected Output Schema

```json
{
    "timestamp": "2026-01-15T09:30:00",
    "config": {"tps": 500, "duration": 60},
    "results": {
        "total_ticks": 30000,
        "processed_ticks": 30000,
        "dropped_ticks": 0,
        "peak_lag_ms": 4.2,
        "avg_lag_ms": 1.8,
        "p99_lag_ms": 3.9,
        "memory_start_mb": 150,
        "memory_end_mb": 185
    },
    "invariant_violations": [],
    "verdict": "PASS"
}
```

---

## 🎯 Delivery Instructions for Jules

1. Create `app/tests/stress_tester.py` following this spec
2. Use **NumPy only** (no Pandas in hot path)
3. **No placeholders** - production-ready code
4. Submit as PR to `feat/stress-test` branch
5. Include unit tests for `TickBurstGenerator`

---

> **Antigravity Sign-off**: This spec is complete and ready for Jules.
