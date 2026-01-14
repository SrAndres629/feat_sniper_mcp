# MASTER PROMPT - PHASE 14: FINAL CALIBRATION & LIVE DEPLOYMENT
## Para: Jules (Architect Protocol)
## De: Antigravity Senior Engineer 1
## Fecha: 2026-01-14

---

## CONTEXTO

Se ha completado la Fase 13 "The Profit Pulse" con los siguientes logros:

### Arquitectura Actual
- **Vector FEAT:** 5D (L1_Mean, L1_Width, L4_Slope, Div_L1_L2, Vol_ZScore)
- **Latencia:** 1.35ms P99 (certificada)
- **Top Predictor:** L4_Slope (pendiente gravitacional)
- **Modo Actual:** SHADOW (piloto de 24h en progreso)

### Módulos Implementados (10/10)
1. ✅ EV Auditor - Calcula Expected Value y Profit Factor
2. ✅ Exhaustion Exit - Salidas inteligentes basadas en física
3. ✅ HUD Upgrade - Telemetría institucional (REGIME, AI_CONF, LAG)
4. ✅ ZMQ Resilience - Pruebas de pánico con Circuit Breaker
5. ✅ Volume Z-Score - 5ta dimensión del vector
6. ✅ Session Filter - Filtrado por Asia/London/NY
7. ✅ Drift Monitor - Detección de degradación del modelo
8. ✅ Compounding - Escalado inteligente de capital
9. ✅ Feature Importance - Análisis de contribución de features
10. ✅ Live Trading Gate - Certificación final para producción

---

## OBJETIVO FASE 14

Preparar el sistema para **LIVE TRADING** certificado con ganancias por hora.

---

## SOLICITUD

Jules, diseña un **PLAN DE 5 MÓDULOS** para la Fase 14 que complete la transición de SHADOW a LIVE:

### Áreas Prioritarias
1. **Calibración del Modelo:** El Feature Importance detectó que L1_Width y Div_L1_L2 son weak features. ¿Debemos reentrenar el modelo excluyéndolas?

2. **Validación Estadística:** El piloto Shadow está en progreso. Necesitamos un módulo que valide estadísticamente que los resultados no son ruido.

3. **Risk Parity:** El Compounding escala capital, pero necesitamos integrar position sizing basado en volatilidad del activo.

4. **Trade Journal:** Registro detallado de cada trade para análisis post-mortem y mejora continua.

5. **Deployment Gate:** Criterios finales para pasar de SHADOW a LIVE con control gradual de exposición.

---

## FORMATO DE RESPUESTA ESPERADO

Para cada módulo, proporciona:
- **Nombre del Módulo**
- **Objetivo**
- **Archivos a crear/modificar**
- **Lógica principal**
- **Criterios de éxito**

Comienza con el **MÓDULO 1** detallado para implementación inmediata.
