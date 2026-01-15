# TRIAD Prompts - Ready to Execute

## 📋 Prompt for Jules (Copy & Paste)

```
Create `app/tests/stress_tester.py` for the FEAT Sniper project.

## Context
This is the NY Open Stress Test to validate system latency under market open conditions.

## Requirements

1. **TickBurstGenerator class**:
   - Generate 500 ticks/second using NumPy (NO Pandas)
   - Configurable volatility multiplier (default 3.0)
   - Configurable spread expansion (default 2.5x)
   - Formula: `price_delta = np.random.normal(0, base_volatility * multiplier)`

2. **PipelineProfiler class**:
   - Measure latency at each FEAT stage (Form, Space, Acceleration, Time)
   - Track peak_lag_ms, avg_lag_ms, p99_lag_ms
   - Memory monitoring (start vs end RSS)

3. **InvariantValidator class**:
   - Assert: form_score is not None
   - Assert: fvg_detected is not None  
   - Assert: acceleration >= -1.0
   - Assert: signal_latency_ms < 5

4. **ReportGenerator class**:
   - Output JSON to logs/stress_test_YYYYMMDD.json
   - Include verdict: PASS if peak_lag_ms < 5ms, FAIL otherwise

5. **CLI interface**:
   - Args: --tps, --duration, --volatility, --output
   - Default: 500 tps, 60s duration, 3.0 volatility

## Constraints
- Production-ready code, NO placeholders
- NumPy only in hot path
- Must integrate with existing ZMQ bridge pattern
- Follow ARCHITECTURE.md laws

## Deliverable
Complete PR to branch `feat/stress-test` with unit tests.
```

---

## 📋 Prompt for OpenCode CLI (After Jules PR is merged)

```powershell
# Step 1: Pull Jules's changes
cd C:\Users\acord\OneDrive\Desktop\Bot\feat_sniper_mcp
git pull origin feat/stress-test

# Step 2: Run the stress test
python -m app.tests.stress_tester --tps 500 --duration 60 --volatility 3.0

# Step 3: Analyze results
python -c "
import json
import glob
files = glob.glob('logs/stress_test_*.json')
if files:
    r = json.load(open(sorted(files)[-1]))
    print(f'Peak: {r[\"results\"][\"peak_lag_ms\"]}ms')
    print(f'Avg: {r[\"results\"][\"avg_lag_ms\"]}ms')
    print(f'Verdict: {r[\"verdict\"]}')
"

# Step 4: Report back to Antigravity
# If PASS: Ready for Shadow Mode
# If FAIL: Share logs for recalibration
```

---

## Status
- [x] Antigravity: Spec complete
- [ ] Jules: Awaiting implementation
- [ ] OpenCode CLI: Awaiting Jules PR
