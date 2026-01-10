# 🦅 FEAT NEXUS PRIME

<div align="center">

![NEXUS](https://img.shields.io/badge/Architecture-NEXUS_PRIME-gold?style=for-the-badge&logo=opsgenie&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MT5](https://img.shields.io/badge/MT5-Build_4000+-orange?style=for-the-badge)
![Supabase](https://img.shields.io/badge/Supabase-Institutional-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)

**The first institutional-grade Algorithmic Trading Platform that bridges MetaTrader 5 with an Autonomous AI Córtex (Dockerized Brain + SSE + RAG).**

[Mission](#-the-mission) • [Command Center](#-command-center) • [Architecture](#-architecture-nexus-prime) • [Omni-Audit](#-omni-audit-self-healing) • [Institutional DB](#-institutional-persistence)

</div>

---

## 🎯 The Mission
NEXUS PRIME is not just an indicator or a bot; it is a **Mission-Critical Ecosystem** designed for high-frequency data extraction, ML-driven analysis, and autonomous decision-making using the FEAT (Form, Space, Acceleration, Time) methodology.

## 🎮 Command Center (One-Switch Control)
The entire system lifecycle is controlled via a unified Python Orchestrator:

- **`nexus.bat`**: Initializes the "Golden Start".
  1. Bootstraps MetaTrader 5.
  2. Orchestrates Docker Compose (Brain + Dashboard).
  3. Executes Internal Sync Audit.
  4. Runs the **War Room Report**.
  5. Opens the Visual Interface.
- **`stop_nexus.bat`**: Executes a Graceful Shutdown.
  1. Safe Docker teardown (Persistence guaranteed).
  2. Professional MT5 process termination.
- **`check_system.bat`**: The Omni-Auditor on demand.

## 🏗️ Architecture: NEXUS PRIME
The system operates on a decentralized core-peripheral model:

1.  **THE BRIDGE (MQL5)**: Native indicators (`UnifiedModel_Main.mq5`, `InstitutionalPVP.mq5`) extraction engine with high-performance ZMQ streaming.
2.  **THE BRAIN (Docker/Python)**: A specialized MCP Server hosting the ML Engine, Data Collectors, and the RAG Memory (ChromaDB).
3.  **THE PERSISTENCE (Supabase)**: Institutional-grade schema for high-frequency tick logging, signal auditing, and model performance tracking.
4.  **THE VISION (Dashboard)**: Next.js + Framer Motion visual cockpit for real-time monitoring.

## 🕵️ Omni-Audit (Self-Healing)
Every component is monitored by the **NEXUS Omni-Auditor**. If a data blocker is detected (MT5 Offline, ZMQ Latency > 50ms, DB Sync Failed), the system generates a machine-readable **REPAIR_REQUEST** to trigger autonomous self-healing by the Antigravity Agent.

## 📊 Institutional Persistence
The database schema (`institutional_schema.sql`) is designed for senior quantitative analysis:
- **`market_ticks`**: Nanosecond-aware price capture.
- **`feat_signals`**: Comprehensive audit trail with confidence scores and top-drivers.
- **`ml_inference_logs`**: Feature snapshots at the moment of prediction for retrospective learning.
- **`knowledge_base`**: RAG persistence for the autonomous agent.

---

## ⚡ Quick Start (The NEXUS Protocol)

### 1. Requirements
- Windows 11 (for MT5 Terminal).
- Docker Desktop.
- Python 3.12+.

### 2. Configuration
Fill your `.env` with the Supabase credentials and MT5 paths:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
MT5_PATH=C:\Program Files\LiteFinance MT5 Terminal\terminal64.exe
```

### 3. Ignition
Run the main orchestrator:
```powershell
.\nexus.bat
```

---

## 📜 Repository Hygiene
This project follows strictly the **Saneamiento Protocol**:
- No binary files in Git.
- No local logs in Git.
- No environment variables in Git.
- Clean development/production branch harmonization.

---

<div align="center">

**Built for the Next Generation of Quantitative Trading.**  
*Powered by SrAndres629 & Antigravity Agent.*

</div>
