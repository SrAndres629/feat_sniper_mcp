
# FEAT NEXUS: AI AGENT OPERATIONAL GUIDE

## 1. System Vision
FEAT NEXUS is an institutional-grade algorithmic trading ecosystem. It uses a multi-stage pipeline (T-F-E-A-S) to synthesize market physics into actionable neural tensors.

## 2. Core Modules
- **`app/skills/feat_sensory.py`**: The 'Nervous System'. Calculates MSS-5 Tensors (Inertia, Entropy, Mass Flow, Acceptance).
- **`app/skills/feat_chain.py`**: The 'Cortex'. Orchestrates the validation stages and final decision fusion.
- **`app/core/auditor_senior.py`**: The 'Black Box'. Records every execution trace in `logs/audit/neural_traces.jsonl`.
- **`app/ml/data_collector.py`**: The 'Memory'. Pipelines market data into standardized feature vectors.

## 3. Kinetic Intent Validation (KIV)
Agents should use the `feat_analyze_acceptance` tool to distinguish between:
- **Structural Conquest**: Solid candle closure (Body > 60% of Range). Indicates true directional intent.
- **Liquidity Probe**: Long wicks (Body < 30% of Range). Indicates rejection or induction (fakeouts).

## 4. Maintenance & Audit
Always run `feat_deep_audit` after any logic modification to ensure sensor alignment and database integrity.

## 5. Neural State Tensors
Features like `momentum_kinetic_micro` and `wick_stress` are normalized vectors. Do not feed raw price data without these abstractions.
