# 🐍 Python Pipeline - ML Backend

> **Motor de Machine Learning y Optimización**  
> *Bayesian Search + SQLite Persistence + Visualización*

---

## 🎯 Propósito

El backend Python procesa los datos exportados por MT5 para:
1. **Entrenar modelos ML** para clasificación de estados
2. **Optimizar umbrales** con búsqueda bayesiana (Optuna)
3. **Persistir histórico** en SQLite
4. **Generar dashboards** HTML interactivos

---

## 🗂️ Estructura de Archivos

```
Python/
├── run_pipeline.py          # 🎯 Orquestador principal
├── db_engine.py             # 💾 Motor SQLite
├── ml_engine.py             # 🧠 Clasificador FSM
├── optuna_optimizer.py      # 🔬 Optimización Bayesiana
├── stats_engine.py          # 📊 Motor estadístico
├── viz_engine.py            # 📈 Generador de dashboards
├── validator.py             # ✅ Validación de configuración
├── brute_force.py           # 🔧 Optimización bruta (legacy)
├── institutional_bridge.py  # 🌉 Bridge HTTP con MT5
│
├── requirements.txt         # 📦 Dependencias
├── bridge_config.json       # ⚙️ Configuración del bridge
├── start_bridge.bat         # 🚀 Script de inicio (Windows)
│
├── unified_model.db         # 💾 Base de datos SQLite
├── fsm_model.joblib         # 🧠 Modelo entrenado
├── optuna_calibration.json  # 📐 Umbrales optimizados
├── ml_thresholds.json       # 📐 Umbrales ML
├── validation_report.json   # 📋 Reporte de validación
│
├── mock_data.csv            # 🧪 Datos de prueba
├── dashboard.html           # 📊 Dashboard generado
└── pipeline.log             # 📝 Log de ejecución
```

---

## 🚀 Quick Start

### 1. Instalar Dependencias
```bash
cd Python/
pip install -r requirements.txt
```

### 2. Ejecutar Pipeline Completo
```bash
python run_pipeline.py --input mock_data.csv --symbol EURUSD --tf H1
```

### 3. Solo Optimización
```bash
python optuna_optimizer.py
```

---

## 📦 Módulos

### 1. run_pipeline.py - Orquestador Principal

```python
"""
Ejecuta el pipeline completo:
1. Ingest Data (CSV -> DB)
2. Train/Update Models (ML Engine)
3. Optimize Thresholds (Optuna)
4. Visualize Results (Viz Engine)
"""

def main():
    # 1. Load and validate CSV
    df = load_and_validate_csv(args.input)
    
    # 2. Ingest to database
    ingest_to_db(db, df, args.symbol, args.timeframe)
    
    # 3. Run ML pipeline
    classifier = run_ml_pipeline(df, output_dir)
    
    # 4. Run optimization
    opt_results = run_optimization_pipeline(df, output_dir, symbol, tf)
    
    # 5. Generate visualization
    run_viz_pipeline(df, classifier, opt_results, output_dir, symbol, tf)
```

**Argumentos CLI:**
| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--input` | `mock_data.csv` | Archivo CSV de entrada |
| `--symbol` | `EURUSD` | Símbolo del instrumento |
| `--tf` | `H1` | Timeframe |
| `--output` | `.` | Directorio de salida |

---

### 2. db_engine.py - Motor SQLite

```python
class UnifiedModelDB:
    """
    SQLite database for Unified Model state history and calibration.
    
    Tables:
    - state_history: All state observations
    - transitions: State transition events
    - calibrations: Threshold configurations
    """
```

**Tablas del Schema:**

#### state_history
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `id` | INTEGER | Primary key |
| `timestamp` | DATETIME | Momento de observación |
| `symbol` | TEXT | Símbolo (EURUSD, etc.) |
| `timeframe` | TEXT | Timeframe (H1, M5, etc.) |
| `state` | TEXT | Estado FSM detectado |
| `confidence` | REAL | Confianza (0-100) |
| `effort` | REAL | Métrica de esfuerzo |
| `result` | REAL | Métrica de resultado |
| `compression` | REAL | Compresión EMA |
| `slope` | REAL | Pendiente normalizada |
| `speed` | REAL | Velocidad de precio |
| `feat_score` | REAL | Score FEAT consolidado |

#### transitions
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `from_state` | TEXT | Estado anterior |
| `to_state` | TEXT | Estado nuevo |
| `confidence` | REAL | Confianza de transición |
| `reason` | TEXT | Razón del cambio |

#### calibrations
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `thresholds_json` | TEXT | Umbrales en JSON |
| `score` | REAL | Score de optimización |
| `method` | TEXT | Método usado (optuna, brute_force) |
| `is_active` | BOOLEAN | ¿Es la calibración activa? |

**API Principal:**
```python
db = UnifiedModelDB("unified_model.db")

# Logging
db.log_state(symbol, timeframe, state, confidence, metrics)
db.log_transition(symbol, tf, from_state, to_state, confidence, reason)

# Queries
history = db.get_state_history(symbol, tf, start_time, end_time)
distribution = db.get_state_distribution(symbol, tf, days=30)
matrix = db.get_transition_matrix(symbol, tf, days=30)

# Calibration
db.save_calibration(symbol, tf, thresholds, score, method)
active = db.get_active_calibration(symbol, tf)

# Export
db.export_to_csv("state_history", "history.csv", symbol, tf)
```

---

### 3. optuna_optimizer.py - Optimización Bayesiana

```python
class OptunaOptimizer:
    """
    Bayesian optimization for FSM thresholds using Optuna.
    
    Advantages over brute force:
    1. Intelligent sampling with TPE (Tree-structured Parzen Estimator)
    2. Early pruning of unpromising trials
    3. ~30-100x faster than grid search
    """
```

**Parámetros Optimizados:**
```python
# Effort thresholds
effort_p20 = trial.suggest_float("effort_p20", 0.1, 0.5)
effort_p80 = trial.suggest_float("effort_p80", 0.8, 2.0)

# Result thresholds
result_p20 = trial.suggest_float("result_p20", 0.1, 0.5)
result_p80 = trial.suggest_float("result_p80", 0.5, 1.5)

# Layer thresholds
layer_sep = trial.suggest_float("layer_sep", 0.5, 3.0)
bias_slope = trial.suggest_float("bias_slope", 0.1, 0.5)
```

**Configuración:**
```python
@dataclass
class OptimizationConfig:
    n_trials: int = 100           # Número de trials
    timeout: Optional[int] = None # Timeout en segundos
    n_startup_trials: int = 10    # Trials aleatorios iniciales
    n_warmup_steps: int = 5       # Steps de warmup
    seed: int = 42                # Seed para reproducibilidad
    show_progress: bool = True    # Mostrar barra de progreso
```

**Uso:**
```python
optimizer = OptunaOptimizer()
optimizer.set_data(effort, result, compression, slope, speed)
best_thresholds = optimizer.optimize()
optimizer.export_calibration("optuna_calibration.json", "EURUSD", "H1")
```

**Output (optuna_calibration.json):**
```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "optimization_method": "optuna",
  "trials": 100,
  "best_score": 0.847,
  "thresholds": {
    "effort_p20": 0.32,
    "effort_p80": 1.45,
    "result_p20": 0.28,
    "result_p80": 0.92,
    "layer_sep": 2.1,
    "bias_slope": 0.25
  }
}
```

---

### 4. ml_engine.py - Clasificador FSM

```python
class FSMClassifier:
    """
    Machine Learning classifier for market states.
    Uses Random Forest with feature engineering.
    """
    
    def train(self, X, y):
        # Feature engineering
        X_eng = self._engineer_features(X)
        
        # Train classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_eng, y)
    
    def predict(self, X):
        X_eng = self._engineer_features(X)
        return self.model.predict(X_eng)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)
```

**Features de Entrada:**
| Feature | Descripción |
|---------|-------------|
| `effort` | Volumen normalizado |
| `result` | Movimiento de precio / ATR |
| `compression` | Compresión de capa Micro |
| `slope` | Pendiente de capa Operational |
| `speed` | Gap Micro-Oper / ATR |
| `rsi` | Valor RSI |
| `macd_hist` | Histograma MACD |

**Output:**
- `fsm_model.joblib` - Modelo serializado

---

### 5. viz_engine.py - Generador de Dashboards

```python
class VizEngine:
    """
    Generates comprehensive HTML dashboards with:
    - State distribution charts
    - Transition heatmaps
    - Performance metrics
    - Calibration history
    """
```

**Gráficos Generados:**
1. **State Distribution** - Pie chart de estados
2. **Transition Heatmap** - Matriz de transiciones
3. **Time Series** - Estados en el tiempo
4. **Effort vs Result Scatter** - Distribución de métricas
5. **Calibration History** - Evolución de scores

**Output:**
- `dashboard.html` - Dashboard interactivo (~5MB)

---

## 📊 Flujo de Datos

```mermaid
sequenceDiagram
    participant MT5 as MetaTrader 5
    participant CSV as CSV File
    participant Pipe as run_pipeline.py
    participant DB as SQLite
    participant ML as ml_engine.py
    participant Opt as optuna_optimizer.py
    participant Viz as viz_engine.py
    
    MT5->>CSV: Export via CInterop
    CSV->>Pipe: Load data
    Pipe->>DB: Ingest records
    DB->>ML: Query training data
    ML->>ML: Train classifier
    ML->>Pipe: Return model
    DB->>Opt: Query metrics
    Opt->>Opt: Bayesian search
    Opt->>Pipe: Return best thresholds
    Pipe->>Viz: Generate dashboard
    Viz->>Viz: Create HTML
    Opt->>MT5: Export calibration JSON
```

---

## ⚙️ Configuración

### requirements.txt
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
optuna>=3.0.0
plotly>=5.0.0
joblib>=1.2.0
```

### bridge_config.json
```json
{
  "host": "127.0.0.1",
  "port": 8888,
  "db_path": "unified_model.db",
  "export_path": "../",
  "log_level": "INFO"
}
```

---

## 🧪 Testing

```bash
# Run with mock data
python run_pipeline.py --input mock_data.csv

# Check pipeline.log for execution details
tail -f pipeline.log

# Validate output
python validator.py --config optuna_calibration.json
```

---

## ⚠️ Notas Importantes

> [!WARNING]
> El archivo `dashboard.html` puede pesar ~5MB debido a los gráficos embebidos.
> Para dashboards más livianos, usar `viz_engine.py` con `embed_data=False`.

> [!TIP]
> Para calibración rápida, usa `run_quick_optimization()` con `n_trials=50`.

> [!NOTE]
> Los modelos se guardan en formato `joblib` para compatibilidad con scikit-learn.

---

*Módulo: Python Pipeline*  
*Versión: 2.0*
