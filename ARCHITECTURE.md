# FEAT Sniper Architecture

> **Source of Truth** — Only Antigravity may modify this file.

## System Laws

### 1. Latency Invariants
- `peak_lag_ms` MUST be < 5ms under normal conditions
- `avg_lag_ms` MUST be < 2ms for signal emission
- Hot path MUST use NumPy, never Pandas

### 2. FEAT Pipeline Order
```
TICK → Form → Space → Acceleration → Time → Signal
```
No step may be skipped. Each must pass validation.

### 3. Risk Boundaries
- CircuitBreaker activates at 2%, 4%, 6% drawdown
- VolatilityGuard halts if ATR > 300% average
- SpreadFilter blocks if spread > 3x symbol average

### 4. ML Constraints
- ValAcc must remain ≈ 0.51 (no data leakage)
- Gating Neuron must align Physics + LSTM
- Hurst exponent calibration: H < 0.4 = mean-revert, H > 0.6 = trend

### 5. Agent Permissions
| Agent        | Read | Write Core | Write Scripts | Modify This File |
| ------------ | ---- | ---------- | ------------- | ---------------- |
| Antigravity  | ✅    | ✅          | ✅             | ✅                |
| Jules        | ✅    | ✅ (via PR) | ❌             | ❌                |
| OpenCode CLI | ✅    | ❌          | ✅             | ❌                |

---

## Current Phase
**Phase 14**: Shadow Mode Preparation  
**Target**: Alpha Robustness Score ≥ 9.0

## Active Constraints
- No live trading until Shadow Mode validation
- All changes require stress test verification
