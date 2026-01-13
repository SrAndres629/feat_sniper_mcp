---
description: Protocolo de colaboración PEER-TO-PEER entre Antigravity y Gemini CLI
---

# Dual-Core Agent Protocol (Antigravity + Gemini CLI)

## Relación Operativa: SENIOR-TO-SENIOR

No tratar al CLI como junior. Ambos son expertos en sus dominios.

## File Locking (Anti-Step)

| Antigravity (Estructura) | Gemini CLI (Cálculo Puro) |
|--------------------------|---------------------------|
| `config.py` | `indicators.py` |
| `nexus_control.py` | `risk_engine.py` |
| `docker-compose.yml` | `hybrid_model.py` |
| `mcp_server.py` | `FEAT_Visualizer.mq5` |

## Ejecución Paralela

```powershell
# Ejemplo: Gemini genera módulo completo
gemini prompt "Actúa como Senior Quant Dev. Genera..." > app/skills/indicators.py
```

## Flujo de Trabajo

1. **Antigravity** prepara configs e imports
2. **Gemini CLI** genera lógica matemática pesada
3. **Antigravity** integra y valida
4. **Ambos** no tocan archivos del otro hasta fusión

## Comando de Verificación de Quota

```powershell
# Si Gemini CLI falla por quota, Antigravity asume la tarea
gemini prompt "status"
```
