# Database Status - FEAT Sniper NEXUS

## 🔧 Current Configuration

| Parameter | Value |
|-----------|-------|
| **Engine** | SQLite 3.x |
| **Mode** | WAL (Write-Ahead Logging) |
| **Path** | `data/market_data.db` |
| **Concurrency** | ✅ Multi-reader + Single-writer |

### PRAGMA Settings
```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;
```

---

## 📊 Schema

### Table: `ticks` (Raw Data)
```sql
CREATE TABLE ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    close REAL, open REAL, high REAL, low REAL, volume REAL,
    rsi REAL, ema_fast REAL, ema_slow REAL, ema_spread REAL,
    feat_score REAL, fsm_state REAL, atr REAL, compression REAL,
    liquidity_above REAL, liquidity_below REAL,
    label INTEGER DEFAULT NULL,
    labeled_at TEXT DEFAULT NULL
);

CREATE INDEX idx_ticks_symbol_ts ON ticks(symbol, timestamp DESC);
CREATE INDEX idx_ticks_unlabeled ON ticks(label) WHERE label IS NULL;
```

### Table: `training_samples` (Labeled)
```sql
CREATE TABLE training_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_id INTEGER REFERENCES ticks(id),
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    -- ... all feature columns ...
    label INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_ts ON training_samples(timestamp);
```

---

## ⚡ Performance Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `BATCH_SIZE` | 100 | Commits each 100 ticks |
| `cache_size` | 10000 | 10MB cache in memory |
| `synchronous` | NORMAL | Balance speed/safety |
| `temp_store` | MEMORY | Temp tables in RAM |

---

## 🛣️ Migration Roadmap

### Phase 1: SQLite WAL (Current) ✅
- Concurrencia básica
- Batch inserts
- Índices optimizados

### Phase 2: PostgreSQL (Future)
```yaml
# docker-compose.yml addition
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
```

### Phase 3: TimescaleDB Hypertables
```sql
-- Particionamiento automático por tiempo
SELECT create_hypertable('ticks', 'timestamp');
```

### Phase 4: pgvector (Unify RAG + Relational)
```sql
CREATE EXTENSION vector;
ALTER TABLE ticks ADD COLUMN embedding vector(384);
```

---

## 📈 Metrics (TBD)

| Metric | Target | Current |
|--------|--------|---------|
| Write Latency | <10ms | Pending |
| Read Latency | <5ms | Pending |
| Concurrent Writers | 1 | 1 (WAL limit) |
| Concurrent Readers | ∞ | Unlimited |

---

*Last updated: 2026-01-10*
