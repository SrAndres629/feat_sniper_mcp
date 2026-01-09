# 📊 07_VISUAL_HUD: FEAT SNIPER HUD

## 📝 Propósito
El **FEAT SNIPER HUD** es la interfaz visual de alta fidelidad diseñada para transformar datos técnicos complejos en decisiones de trading instantáneas e intuitivas. Inspirado en los tableros de instrumentos de aviación, prioriza la claridad jerárquica y la respuesta emocional mediante colores.

---

## 🏗️ Layout del Dashboard

El HUD se posiciona en la esquina superior izquierda (`CORNER_LEFT_UPPER`) y se divide en dos bloques principales:

### 1. El Núcleo (FEAT SCORE)
Es el indicador principal de confianza del sistema.
- **Componente**: `HUD_BG` + `HUD_ScoreNum`.
- **Lógica de Color**:
  - **Verde Sniper (0,180,60)**: Score > 70 → **COMPRA FUERTE**.
  - **Rojo Sniper (220,40,40)**: Score < 30 → **VENTA FUERTE**.
  - **Amarillo Rango (100,100,0)**: Score 50-69 → **OBSERVAR**.
  - **Gris (80,80,80)**: Score Neutral / Inactivo.

### 2. Panel de Diagnóstico (Satélites)
Ubicado debajo del Score, proporciona el contexto necesario para validar la señal.

| Atributo | Fuente de Datos | Significado |
|----------|----------------|-------------|
| **MODO** | `CFSM::GetStateString()` | Estado actual del ciclo Wyckoff. |
| **KILLZONE** | `CFEAT::GetTime()` | Estado binario (ON/OFF) basado en la sesión. |
| **VELOCIDAD** | `CFEAT::GetAccel()` | Barra de progreso que muestra el momentum actual. |
| **INSTITUCIONAL** | `SSniperReport::isInstitutional` | Alerta de velas de alta intención (Aceleración > 1.2 ATR). |

---

## 🎨 Guía de Colores Institucional

- **Fondo Primario**: `C'20,20,20'` (Anthracite suave para evitar fatiga visual).
- **Bordes**: `C'60,60,60'` (Gris técnico).
- **Textos Secundarios**: `clrSilver` (Plata).
- **Alertas**: `clrGold` (Institucional / Importante).
- **Velocidad Activa**: `clrAqua`.

---

## 🛠️ Public API (`CVisuals.mqh`)

### `void Init(string prefix, long chartID)`
Inicializa el sistema de objetos en el gráfico especificado.

### `void SetComponents(CEMAs* e, CFEAT* f, CLiquidity* l, CFSM* sm)`
Inyecta los punteros de los motores de cálculo para la extracción de datos en tiempo real.

### `void Draw(datetime t, double close)`
El motor de renderizado principal.
1. Elimina objetos obsoletos del frame anterior (`HUD`).
2. Extrae métricas de `FEAT` y `FSM`.
3. Calcula estados de color.
4. Redibuja los objetos `OBJ_RECTANGLE_LABEL` y `OBJ_LABEL`.

---

## 🚀 Optimización de Rendimiento
- **Gestión de Objetos**: Se utiliza un prefijo (`m_prefix`) para evitar colisiones con otros indicadores.
- **Redibujado Selectivo**: El HUD solo se actualiza cuando `Draw()` es llamado (generalmente en cada Tick en `UnifiedModel_Main.mq5`).
- **Separación de PVP**: El mapa de volumen pesado se delega a `InstitutionalPVP.mq5` para mantener el HUD fluido a >60 FPS (virtuales).

---

## 📋 Notas de Implementación
- Se requiere la fuente **"Impact"** instalada en el sistema para el Score numérico grande.
- Se recomienda usar un fondo de gráfico oscuro (`clrBlack`) para máximo contraste.
