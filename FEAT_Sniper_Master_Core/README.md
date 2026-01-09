<p align="center">
  <img src="https://img.shields.io/badge/Platform-MetaTrader%205-blue?style=for-the-badge" alt="Platform"/>
  <img src="https://img.shields.io/badge/Language-MQL5%20%7C%20Python-green?style=for-the-badge" alt="Language"/>
  <img src="https://img.shields.io/badge/Version-2.0-orange?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/License-Private-red?style=for-the-badge" alt="License"/>
</p>

# 🎯 FEAT_Sniper_Master_Core

> **Sistema Cuántico de Trading Institucional**  
> *Análisis Multifractal con Machine Learning para MetaTrader 5*

---

## 📋 Descripción

**FEAT_Sniper_Master_Core** es un indicador avanzado de MT5 que implementa un sistema de análisis institucional de alta precisión. Combina:

- 🔢 **31 Capas EMA Multifractal** - Análisis de tendencias a múltiples escalas
- 💧 **Detección de Liquidez** - Mapeo de zonas institucionales (FVG, OB, BOS/CHoCH)
- ⚙️ **Máquina de Estados** - Clasificación Wyckoff automática
- 🧠 **Pipeline de ML** - Optimización bayesiana con Optuna
- 📊 **HUD Profesional** - Dashboard en tiempo real

---

## 🏛️ Arquitectura

```
FEAT_Sniper_Master_Core/
├── 📊 UnifiedModel_Main.mq5       # Indicador principal
├── 📊 InstitutionalPVP.mq5        # Volume Profile
│
├── 📦 Include/UnifiedModel/
│   ├── CEMAs.mqh                  # Motor 31 EMAs
│   ├── CFEAT.mqh                  # Inteligencia FEAT
│   ├── CLiquidity.mqh             # Mapeo liquidez
│   ├── CFSM.mqh                   # State Machine
│   ├── CVisuals.mqh               # HUD Dashboard
│   ├── CInterop.mqh               # Bridge Python
│   └── CMultitemporal.mqh         # Multi-TF
│
├── 🐍 Python/
│   ├── run_pipeline.py            # Orquestador ML
│   ├── optuna_optimizer.py        # Bayesian Search
│   ├── db_engine.py               # SQLite persistence
│   └── viz_engine.py              # Dashboard HTML
│
└── 📚 docs/mapa_conceptual/       # Documentación modular
```

---

## ⚡ Quick Start

### 1. Instalación en MT5

```bash
# Copiar archivos a la carpeta de MT5
xcopy /E /I "Include\UnifiedModel" "%APPDATA%\MetaQuotes\Terminal\<ID>\MQL5\Include\UnifiedModel"
copy "UnifiedModel_Main.mq5" "%APPDATA%\MetaQuotes\Terminal\<ID>\MQL5\Indicators\"
```

### 2. Compilación

Abrir MetaEditor → Abrir `UnifiedModel_Main.mq5` → Compilar (F7)

### 3. Uso

Arrastrar el indicador `UnifiedModel_Main` a cualquier chart.

---

## 🐍 Pipeline Python

```bash
cd Python/
pip install -r requirements.txt
python run_pipeline.py --input mock_data.csv --symbol EURUSD --tf H1
```

---

## 🎪 Sistema de Capas EMA

| Capa | Metáfora | Color | Períodos |
|------|----------|-------|----------|
| **MICRO** | 🌬️ Gas | Rojo/Amarillo | 1-12 |
| **OPERATIONAL** | 🌊 Agua | Naranja | 16-224 |
| **STRUCTURAL** | 🧱 Muro | Verde | 50+ |
| **MACRO** | 📍 Magneto | Azul | 256-1280 |
| **BIAS** | ⚖️ Régimen | Gris | 2048 |

---

## 🧠 Cadena de Decisión

```
🔧 ENGINEER          🎯 TACTICIAN         🎖️ SNIPER
   (Análisis)    →      (Contexto)     →    (Disparo)
   
   • Vectores           • Killzones          • DISPARAR
   • Presión            • POI detectado      • ABORTAR
   • Energía            • Premium/Discount   • Entry/SL/TP
```

---

## 📊 FEAT SNIPER HUD (Heads-Up Display)

El sistema utiliza un HUD de combate de alta visibilidad:

- **EL NÚCLEO (SCORE)**: Indicador central grande con color dinámico (Verde >75, Rojo <25).
- **KILLZONE STATUS**: Alerta visual cuando el tiempo es operable.
- **BARRA DE POTENCIA**: Visualización en tiempo real de la aceleración del precio.
- **AVISO INSTITUCIONAL**: Detector de velas de intención de alta velocidad.

---

## 📦 Dependencias

### MQL5
- MetaTrader 5 Build 3000+
- Ninguna biblioteca externa

### Python
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
optuna>=3.0.0
plotly>=5.0.0
```

---

## 📚 Documentación

Documentación completa disponible en `docs/mapa_conceptual/`:

| Archivo | Contenido |
|---------|-----------|
| [00_VISION_GLOBAL](docs/mapa_conceptual/00_VISION_GLOBAL.md) | Visión general |
| [01_MQL5_CORE](docs/mapa_conceptual/01_MQL5_CORE.md) | Arquitectura MQL5 |
| [02_CFEAT_INTELLIGENCE](docs/mapa_conceptual/02_CFEAT_INTELLIGENCE.md) | Sistema FEAT |
| [03_LIQUIDITY_MAPPING](docs/mapa_conceptual/03_LIQUIDITY_MAPPING.md) | Detección liquidez |
| [04_FSM_STATES](docs/mapa_conceptual/04_FSM_STATES.md) | Máquina de estados |
| [05_PYTHON_PIPELINE](docs/mapa_conceptual/05_PYTHON_PIPELINE.md) | Backend Python |
| [06_INTEROPERABILITY](docs/mapa_conceptual/06_INTEROPERABILITY.md) | Bridge MT5↔Python |
| [07_VISUAL_HUD](docs/mapa_conceptual/07_VISUAL_HUD.md) | Dashboard visual |
| [08_ROADMAP](docs/mapa_conceptual/08_ROADMAP.md) | Objetivos futuros |

---

## ⚠️ Disclaimer

> Este software es solo para fines educativos y de investigación.  
> El trading de instrumentos financieros conlleva riesgo significativo.  
> El uso de este sistema es bajo su propia responsabilidad.

---

## 📄 Licencia

Proyecto privado. Todos los derechos reservados.

---

<p align="center">
  <b>FEAT_Sniper_Master_Core</b> • v2.0 • Enero 2026
</p>
