# ML Status - Quantum Leap Protocol

## 🔧 Configuration

| Flag | Value | Description |
|------|-------|-------------|
| `EXECUTION_ENABLED` | `False` | Shadow Mode activo |
| `LOG_LEVEL` | `INFO` | Nivel de logging |
| `SEQ_LEN` | `32` | Longitud de secuencia LSTM |
| `N_LOOKAHEAD` | `10` | Velas para Oracle labeling |
| `PROFIT_THRESHOLD` | `0.002` | 0.2% para WIN |
| `BATCH_SIZE` | `100` | Ticks por batch insert |

## 💾 Database (SQLite WAL)

| Setting | Value |
|---------|-------|
| **Path** | `data/market_data.db` |
| **Mode** | WAL (Write-Ahead Logging) |
| **Concurrency** | ✅ Reads/Writes simultáneas |
| **Tables** | `ticks`, `training_samples` |
| **Indexes** | `symbol+timestamp`, `unlabeled` |

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;
```

## 🧠 Models

### GBM (Gradient Boosting)
- **Path**: `models/gbm_v1.joblib`
- **Status**: ⏳ Awaiting training data
- **Validation**: TimeSeriesSplit (5 folds)
- **Metric**: LogLoss

### LSTM (Attention)
- **Path**: `models/lstm_v1.pt`
- **Status**: ⏳ Awaiting training data
- **Architecture**: Bidirectional LSTM + Self-Attention
- **Validation**: Temporal split 80/20

### Anomaly Detector
- **Type**: IsolationForest
- **Contamination**: 0.01
- **Purpose**: Detect market manipulation

## 📊 Data Pipeline

```
MT5 Candle → data_collector.py → Oracle (N-lookahead) → CSV
                                                          ↓
                                              train_models.py
                                                          ↓
                                               models/*.joblib/.pt
                                                          ↓
                                              ml_engine.py → Predictions
```

## 🛡️ Shadow Mode

- **Estado**: ACTIVADO
- **Comportamiento**: Predice y loguea, NO ejecuta órdenes
- **Log**: `data/shadow_predictions.jsonl`

## 📈 Training Requirements

| Metric | Required | Current |
|--------|----------|---------|
| Samples | 1,000+ | 0 |
| Features | 15 | ✅ |
| Label Rate | N/A | Pending |

## 🚀 Next Steps

1. Run data collection for 24-48h
2. Execute `python app/ml/train_models.py`
3. Verify shadow predictions
4. Review ML_STATUS.md metrics
5. Enable execution: `EXECUTION_ENABLED=True`

---
*Last updated: 2026-01-10*
