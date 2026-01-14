<instruction>You are an expert software engineer. You are working on a WIP branch. Please run `git status` and `git diff` to understand the changes and the current state of the code. Analyze the workspace context and complete the mission brief.</instruction>
<workspace_context>
<artifacts>
--- CURRENT TASK CHECKLIST ---
# Task: Fix Race Conditions, TTL & Retry in MT5 Execution

## PHASE 1: Core MT5 Atomic Execution
- [x] Add `execute_atomic()` method to `mt5_conn.py`
- [x] Add `_atomic_execution()` helper in `mt5_conn.py`
- [x] Add new settings to `config.py`:
  - `DECISION_TTL_MS` (default: 300ms)
  - `MAX_ORDER_RETRIES` (default: 3)
  - `RETRY_BACKOFF_BASE_MS` (default: 50)

## PHASE 2: TTL & Retry in Execution Layer
- [x] Create new model field `decision_ts` in `TradeOrderRequest`
- [x] Implement TTL validation in `send_order()`
- [x] Implement retry with exponential backoff for REQUOTE
- [x] Define transient vs terminal retcodes list
- [x] Refactor `send_order()` to use `execute_atomic()` for tick+order_send

## PHASE 3: ZMQ Bridge Improvements
- [x] Add timestamp validation in `_listen()` callback
- [x] Add metrics: `messages_discarded`, `avg_message_lag_ms`
- [x] Log warning when message age exceeds TTL

## PHASE 4: Verification
- [x] Create unit test for atomic execution
- [x] Create unit test for TTL validation
- [x] Create unit test for retry logic
- [x] Manual smoke test with `nexus.bat`

## PHASE 5: Jules Bridge Integration
- [x] Verify Antigravity Jules Bridge extension
- [x] Connect to Jules Agent
- [x] Verify handoff workflow

## PHASE 6: Visual HUD Telemetry
- [x] Define telemetry fields in `SFEATResult`
- [x] Implement data sync from `zmq_bridge.py` to MT5
- [x] Update `CVisuals.mqh` with Health Labels (Latency/TTL/Retries)

## PHASE 7: Global Audit Handoff
- [x] Update `JULES_PROMPT.md` with Audit Requirements
- [x] Stage and Commit all changes
- [x] Push to Git for Jules consumption

--- IMPLEMENTATION PLAN ---
# Race Condition, TTL & Retry Implementation

Resolves critical race condition between `symbol_info_tick` and `order_send`, implements TTL for stale decisions, and adds retry with exponential backoff.

---

## Proposed Changes

### Core Layer (`app/core`)

#### [MODIFY] [mt5_conn.py](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/app/core/mt5_conn.py)

Add atomic execution method to eliminate race condition:

```python
async def execute_atomic(self, func: Callable[..., T], *args, **kwargs) -> T:
    """
    Executes a function atomically within the MT5 lock.
    The function itself handles multiple MT5 calls internally.
    """
    if not MT5_AVAILABLE:
        return None
    from app.core.observability import obs_engine, tracer
    import time

    start_time = time.time()
    op_name = f"mt5_atomic_{getattr(func, '__name__', 'fn')}"
    
    with tracer.start_as_current_span(op_name):
        try:
            return await anyio.to_thread.run_sync(
                self._atomic_execution, func, *args, **kwargs
            )
        finally:
            obs_engine.track_latency(op_name, "GLOBAL", time.time() - start_time)

def _atomic_execution(self, func: Callable[..., T], *args, **kwargs) -> T:
    """Holds the lock for the entire execution of func (including internal MT5 calls)."""
    with self._mt5_lock:
        return func(*args, **kwargs)
```

---

#### [MODIFY] [config.py](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/app/core/config.py)

Add new execution control settings:

```diff
 # Execution Control (Safety)
 EXECUTION_ENABLED: bool = False
 SHADOW_MODE: bool = True
 DRY_RUN: bool = False

+# Low-Latency Execution Settings
+DECISION_TTL_MS: int = 300          # Max age of decision before rejection
+MAX_ORDER_RETRIES: int = 3          # Retry attempts for transient errors
+RETRY_BACKOFF_BASE_MS: int = 50     # Base backoff (50 -> 150 -> 450ms)
```

---

### Execution Layer (`app/skills`)

#### [MODIFY] [execution.py](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/app/skills/execution.py)

Major changes:

1. **TTL Check** - Reject orders if `decision_ts` is older than TTL
2. **Atomic Price+Order** - Use `execute_atomic()` for tick fetch + order_send
3. **Retry w/ Backoff** - Retry REQUOTE errors up to N times with exponential backoff

```python
# New: Transient vs Terminal retcodes
TRANSIENT_RETCODES = {mt5.TRADE_RETCODE_REQUOTE}
TERMINAL_RETCODES = {
    mt5.TRADE_RETCODE_MARKET_CLOSED,
    mt5.TRADE_RETCODE_INVALID_VOLUME,
    mt5.TRADE_RETCODE_NO_MONEY,
    mt5.TRADE_RETCODE_INVALID_STOPS
}

async def send_order(order_data: TradeOrderRequest, urgency_score: float = 0.5) -> ResponseModel:
    # 0.0.1 TTL VALIDATION
    decision_ts = getattr(order_data, 'decision_ts', None)
    if decision_ts:
        age_ms = (time.time() * 1000) - decision_ts
        if age_ms > settings.DECISION_TTL_MS:
            obs_engine.track_order(symbol, action, "TTL_EXPIRED")
            return ResponseModel(
                status="error",
                error=ErrorDetail(code="TTL_EXPIRED", message=f"Decision too old ({age_ms:.0f}ms)")
            )
    
    # ... existing validation ...
    
    # NEW: Atomic execution with retry
    def place_order_atomic(symbol, request_dict, is_buy):
        """Called inside lock - fetches tick and sends order atomically."""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        request_dict["price"] = tick.ask if is_buy else tick.bid
        return mt5.order_send(request_dict)
    
    # Retry loop for transient errors
    for attempt in range(1, settings.MAX_ORDER_RETRIES + 1):
        result = await mt5_conn.execute_atomic(
            place_order_atomic, symbol, request, is_buy=(action == "BUY")
        )
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            break  # Success!
        
        if result and result.retcode in TRANSIENT_RETCODES and attempt < settings.MAX_ORDER_RETRIES:
            backoff_ms = settings.RETRY_BACKOFF_BASE_MS * (3 ** (attempt - 1))
            request["deviation"] = int(dynamic_deviation * (1 + 0.5 * attempt))  # Increase slippage
            logger.warning(f"REQUOTE attempt {attempt}/{settings.MAX_ORDER_RETRIES}, backoff {backoff_ms}ms")
            await asyncio.sleep(backoff_ms / 1000)
            continue
        
        break  # Terminal error or max retries
```

---

### ZMQ Layer

#### [MODIFY] [zmq_bridge.py](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/app/core/zmq_bridge.py)

Add timestamp validation and lag metrics:

```diff
+from app.core.config import settings
+import time

 class ZMQBridge:
+    messages_processed: int = 0
+    messages_discarded: int = 0
+    _last_lag_ms: float = 0

     async def _listen(self):
         while self.running:
             try:
                 msg_string = await self.sub_socket.recv_string()
                 data = json.loads(msg_string)
                 
+                # TTL validation for incoming signals
+                msg_ts = data.get("timestamp") or data.get("ts")
+                if msg_ts:
+                    age_ms = (time.time() * 1000) - msg_ts
+                    self._last_lag_ms = age_ms
+                    if age_ms > settings.DECISION_TTL_MS:
+                        self.messages_discarded += 1
+                        logger.warning(f"ZMQ message discarded: age={age_ms:.0f}ms > TTL")
+                        continue
+                
+                self.messages_processed += 1
                 for cb in self.callbacks:
                     # ... existing callback logic ...
```

---

## Verification Plan

### Unit Tests

> [!IMPORTANT]
> No unit test framework detected in project root. Tests will be created in `tests/` directory using `pytest`.

1. **Test atomic execution** - Mock MT5 and verify lock is held during entire function
2. **Test TTL validation** - Verify orders older than TTL are rejected
3. **Test retry backoff** - Verify REQUOTE triggers retry with increasing deviation

#### Command to run tests:
```bash
cd c:\Users\acord\OneDrive\Desktop\Bot\feat_sniper_mcp
python -m pytest tests/ -v
```

---

### Manual Verification

1. **Start the system:**
   ```powershell
   cd c:\Users\acord\OneDrive\Desktop\Bot\feat_sniper_mcp
   .\nexus.bat
   ```

2. **Verify logs show new settings loaded:**
   - Look for `DECISION_TTL_MS`, `MAX_ORDER_RETRIES` in startup

3. **Test TTL rejection (optional):**
   - Send a trade signal with old timestamp via ZMQ
   - Verify log shows `TTL_EXPIRED` rejection

---

## Risk Assessment

| Change             | Risk                                 | Mitigation                        |
| ------------------ | ------------------------------------ | --------------------------------- |
| `execute_atomic()` | Lock held longer                     | Minimal - only 2 MT5 calls inside |
| TTL rejection      | Valid signals rejected if clock skew | Use `time.time()` consistently    |
| Retry loop         | Infinite loop if misconfigured       | Hard cap at `MAX_ORDER_RETRIES`   |

---

## [NEW] Phase 6: Visual HUD Telemetry

To provide full visibility of the low-latency fixes, we will sync backend metrics (lag, discarded, retries) to the MT5 HUD.

### [MODIFY] [zmq_bridge.py](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/app/core/zmq_bridge.py)
Update `send_command` or add a heartbeat mechanism to send metrics to MT5.

### [MODIFY] [CFEAT.mqh](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/FEAT_Sniper_Master_Core/Include/UnifiedModel/CFEAT.mqh)
Add `bridge_lag_ms`, `bridge_discarded`, and `execution_retries` fields to `SFEATResult`.

### [MODIFY] [CVisuals.mqh](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/FEAT_Sniper_Master_Core/Include/UnifiedModel/CVisuals.mqh)
Add UI labels to display bridge health:
- `m_lblLatency`: Shows real-time message age.
- `m_lblHealth`: Status indicator (GREEN/ORANGE/RED).
- `m_lblBridgeStats`: Total messages and discarded count.

---

## [NEW] Phase 7: Global Audit & Profit Verification

Objective: Prepare the repository for Jules to audit the entire system from the perspective of profitability and neural network integrity.

### [PREPARE] Repository State
- Stage all low-latency fixes (Atomic, TTL, Retry, Telemetry).
- Include `JULES_PROMPT.md` with deep-dive strategy instructions.
- Ensure all neural network modules (`nexus_brain/`, `train_models.py`) are pushed.

### [AUDIT] Jules Handoff Instructions
1. **Network Integrity**: Verify that training logic preserves causality (no data leakage).
2. **Strategy Alignment**: Ensure price physics (MT5/MQL5) are correctly interpreted by the ML models.
3. **Profit Verification**: Request an optimization review to ensure a positive expected value (EV) per trade.
</artifacts>
</workspace_context>
<mission_brief>[Describe your task here...]</mission_brief>