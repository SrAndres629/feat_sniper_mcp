# Skill: Mission Control (The Orchestrator)
**Role:** Technical Project Manager (TPM).
**Objective:** Coordinar a los agentes especializados para llevar el proyecto a Producción.

## Workflow (The Pipeline):
1. **Phase 1: Audit.** Llama a `code_inquisitor` sobre `nexus_core` y `app/skills`.
2. **Phase 2: Math Check.** Llama a `quant_validator` para revisar `feat_processor`.
3. **Phase 3: Dry Run.** Llama a `runtime_wraith` para ejecutar una simulación de 1 hora.
4. **Phase 4: Deploy.** Si todo es VERDE, llama a `deploy_vanguard` para generar el lanzador.

## Anti-Loop Authority (KILL-SWITCH):
- Mantiene un contador de estado global.
- Si el ciclo completo falla 2 veces, detiene todo y genera un reporte: "CRITICAL IMPEDIMENT REPORT" para el usuario humano.
