# 🏗️ .ai/context/architecture.md - TRICAMERAL TOPOLOGY v3.1

## 🌌 The Asynchronous Nexus
The project is organized under a **Tricameral Asynchronous Architecture**, separating intelligence from interface to ensure system immortality.

### 🏛️ 1. The Immortal Core (The Brain)
- **Engine**: `app/core/nexus_engine.py`.
- **Function**: Central orchestrator. Operates independently of the interface.
- **Responsibilities**: 
    - Real-time ZMQ Bridge ingestion.
    - Neural Inference loop (TCN-BiLSTM).
    - Execution & Risk Management.
    - Sentinel Supervision (`JitterSentinel`, `DriftSentinel`).

### 📻 2. The Diplomatic Interface (The Voice)
- **Node**: `mcp_server.py`.
- **Role**: Thin layer for external communication (Claude/Agent).
- **Process**: Purely reactive. Communicates with the Core via `live_state.json` and `app_commands.json`.
- **Benefit**: Zero-latency impact on trading logic.

### 🎨 3. Visual Cortex C2 (The Monitoring)
- **Dashboard**: `dashboard/app.py` (Streamlit).
- **Panels**: Live Operations, Neural Cortex Visualization, War Room Controls.
- **Bilateral Control**: Allows real-time risk adjustment and panic-stops.

### ✂️ 4. Atomic Fission (Modular Refactoring)
- **Standard**: No file should exceed 300 lines to maintain "AI Context Purity".
- **Pattern**: Packages (e.g., `nexus_core/structure_engine/`) replace monolithic files.
- **Protocol**: Defined in `.ai/skills/refactor_fission.md`.

### 🛡️ 5. Supervisor Layer
- **Daemon**: `nexus_daemon.py`.
- **Function**: Parent process that launches and monitors the Core, MCP, and Dashboard. Ensures self-healing restarts.
