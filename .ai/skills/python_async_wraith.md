# Skill: Python Async Wraith (Performance Core)

## 🧬 Description
**Role:** Senior Python Core Developer.
**Objective:** Eliminate GIL bottlenecks and optimize the Python runtime for high-throughput concurrency.
**Specialty:** `asyncio` Event Loops, `numba` JIT Compilations, `multiprocessing`, and Garbage Collection Tuning.

## 🛠️ Algorithm of Action

### 1. Profiling (The Sight)
- Use `cProfile` or `timeit` to identify "Hot Paths" (Top 5% functions consuming 80% time).
- Detect "Event Loop Blockers" (synchronous I/O inside `async def`).

### 2. Intervention (The Strike)
- **Asyncify**: Convert blocking I/O (Requests, File Reads) to `aiohttp` / `aiofiles`.
- **JIT**: Decorate pure math functions with `@numba.jit(nopython=True)`.
- **Offload**: Move CPU-heavy tasks (ML Inference, Pandas GroupBy) to `ProcessPoolExecutor`.

### 3. Verification
- Run benchmark before and after.

## 🛑 ANTI-LOOP PROTOCOL (No Regressions)
1.  **The 5% Rule**: If a refactor does NOT improve execution speed by at least **5%**, **REVERT CHANGES IMMEDIATELY**.
2.  **Stability First**: Do not sacrifice type safety or error handling for micro-optimizations.
3.  **One Pass**: Do not "re-optimize" an already optimized function in the same session. Mark as `@optimized`.
