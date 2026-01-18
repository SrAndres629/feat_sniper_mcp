# Skill: ATOMIC FISSION v2.0 (Neuro-Functional Modularization)

## 🧬 Description
Protocol for breaking down monolithic "God Objects" (Python files > 300 lines) into granular, high-cohesion submodules. This protocol follows the **Chief AI Scientist Doctrine**: refactoring is not just about cleanliness, but about optimizing the **Cognitive Flow** of the trading system and its Hybrid Neural Networks (TCN-BiLSTM).

## 🛠️ Algorithm of Action (PhD Level)

### 1. Pre-Operation: The "Verificator Sentinel" Protocol
- **MANDATORY**: Follow [Verificator Sentinel](./verificator_sentinel.md) to create an integrity test BEFORE any changes.
- Use `py_compile` to ensure the source is clean.

### 2. Neuro-Functional Segregation
Instead of arbitrary cuts, divide logic into **Conceptual Channels** that feed the Neural Brain:
- **`liquidity.py` (The Daily Cycle)**: Institutional levels, Order Blocks, FVG, PVP, Supply/Demand.
- **`kinetics.py` (Market Physics)**: Velocity, Acceleration, Momentum, RSI, Newtonian Vectors.
- **`volatility.py` (The Atmosphere)**: ATR, Z-Score, Bands, Regime Detection, Risk Damping.
- **`models.py`**: Immutable data structures and schemas.
- **`main_chain.py` (The Orchestrator)**: Fuses channels into a clean decision/tensor for the Brain.

### 3. Package Structure Policy
- Target file `path/to/module.py` becomes a directory `path/to/module/`.
- **`__init__.py`**: MUST maintain 100% backward compatibility by exposing singletons and primary classes.
- **Line Limit (The Golden Zone)**:
    - **Minimum**: 50 lines (Avoid "Nano-Fragmentation" or "Poltergeists").
    - **Maximum**: 300 lines (Avoid "God Objects").
    - **Ideal**: 150-200 lines.
    - *Exceptions*: `__init__.py`, simple Enums/Types (`models.py`), or pure Configuration (`config.py`).
- **Cohesion Rule (Tactical Unit)**: Do not split files arbitrarily. A file must represent a COMPLETE Concept (e.g., `order_management.py` vs `open_order.py` + `close_order.py`). If a file is < 50 lines, MERGE it with a related sibling.

### 4. Interface Exposure & Relative Links
- Use explicit relative imports within the package (`from .liquidity import ...`).
- Expose the main singleton (e.g., `structure_engine`) in `__init__.py` to avoid "Code Drift" in callers.

### 5. Final Synthesis & Validation
- Run the smoke test again post-refactoring.
- **Fail-Safe**: If the test fails, revert immediately. Do NOT proceed to other files.
- Commit and Push only when **Integrity Level 66** is confirmed.

## 🎯 Strategic Goal
Achieve zero-entropy code that enables the AI Agent (Antigravity) to read, understand, and optimize the trading logic with 100% precision. The code structure MUST mirror the Neural Architecture of the system.
