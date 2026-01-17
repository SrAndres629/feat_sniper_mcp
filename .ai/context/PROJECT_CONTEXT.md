# 🧠 PROJECT_CONTEXT.md: The Technical Landscape (v3.1)

## 🚀 Current Architectural State (Level 66 - Immortal)
The system has transitioned from a centralized "God Object" into a **Tricameral Asynchronous Framework**, ensuring zero single-point-of-failure.

### 1. The Cognition Loop (Nexus Engine)
The "Immortal Path" follows this strict hierarchy:
1. **ZMQ Ingestion**: Real-time MT5 tick stream handled in dedicated background loop.
2. **Spatio-Temporal Engineering**: `FEATProcessor` (now vectorized v3.1) creates 50x50 Spatial Maps for any historical depth.
3. **Neural Adaptation**: `MLEngine` generates Multi-Head predictions.
4. **Bayesian Convergence**: Evaluates Neural Alpha vs Kinetic Coherence.
5. **Epistemic Gate**: If Uncertainty > `settings.CONVERGENCE_MAX_UNCERTAINTY`, logic is blocked.

### 2. C2 Command & Control
- **Streamlit Dashboard**: Provides warfare-grade UI for Live Operations and Neural Vision.
- **Bilateral Commands**: Adjusting `RISK_FACTOR` or `PANIC_CLOSE` via `data/app_commands.json` without stopping the core engine.

### 3. Modular Integrity (Atomic Fission)
- **Constraint**: All "God Objects" (>500 lines) are being dismantled into atomic subpackages.
- **Example**: `nexus_core/structure_engine/` is the flagship modular implementation.
- **Stability**: `__init__.py` files ensure backward compatibility while codebase remains lean.

### 4. Institutional Risk & Shield
- **Bayesian Kelly**: Lot sizing proportional to $\frac{Probability}{Uncertainty}$.
- **Performance Damping**: Real-time Profit Factor (from `DriftMonitor`) automatically scales risk.
- **The Vault**: 50% of realized profits are electronically "locked" from the margin.

---

## 🛡️ Critical Constraints for AI (Antigravity Protocol)
- **Precision**: Zero hardcoded magic numbers. Use `app/core/config.py`.
- **Modularity**: New logic MUST be added to specific modules, keeping file size < 300 lines.
- **Process Supervision**: `nexus_daemon.py` is the entry point. It handles all process lifecycles.
- **Fail-Stop**: Critical failures in the `NexusEngine` trigger an immediate shutdown of Trading Operations while keeping the Dashboard and MCP alive for investigation.
