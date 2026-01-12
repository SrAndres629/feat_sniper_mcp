-- =============================================================================
-- FEAT NEXUS INSTITUTIONAL SCHEMA v6.0
-- Protocolo de Inteligencia Multifractal (MIP)
-- =============================================================================

-- 1. CENTRAL MARKET DATA (Multitemporal Feature Hub)
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL, -- M1, M5, M15, M30, H1, H4, D1, W1
    
    -- OHLCV Data
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    
    -- Technical Tensors (Computed per Timeframe)
    rsi REAL,
    atr REAL,
    ema_fast REAL,
    ema_slow REAL,
    fsm_state INTEGER,
    feat_score REAL,
    
    -- Contextual Data
    liquidity_ratio REAL,
    volatility_zscore REAL,
    
    -- ML Meta-info
    label INTEGER DEFAULT NULL,
    labeled_at TIMESTAMP DEFAULT NULL,
    
    UNIQUE(tick_time, symbol, timeframe)
);

-- 2. FRACTAL ANALYSIS REGISTRY
CREATE TABLE IF NOT EXISTS fractal_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    
    hurst_exponent REAL,     -- > 0.5 (Trend), < 0.5 (Mean Revert)
    fractal_dimension REAL,  -- Complexity metric
    regime_mode TEXT,        -- TREND, RANGE, ERRATIC
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. INSTITUTIONAL ACTIVITY LOG
CREATE TABLE IF NOT EXISTS bot_activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event TEXT NOT NULL,
    level TEXT DEFAULT 'INFO', -- INFO, WARNING, ERROR, CRITICAL
    phase TEXT,                -- MIP, GENESIS, EXECUTION
    details JSON,
    trace_id TEXT              -- For correlation across components
);

-- 4. PERFORMANCE TRACKING (Enhanced)
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe_context TEXT,    -- Multi-temporal context at trade
    
    win_rate REAL,
    profit_factor REAL,
    sharpe_ratio REAL,
    drawdown_max REAL,
    
    hyperparameters JSON
);

-- INDEXING FOR HIGH-FREQUENCY RETRIEVAL
CREATE INDEX IF NOT EXISTS idx_market_lookup ON market_data(symbol, timeframe, tick_time DESC);
CREATE INDEX IF NOT EXISTS idx_fractal_lookup ON fractal_analysis(symbol, timeframe, analysis_time DESC);
CREATE INDEX IF NOT EXISTS idx_unlabeled_context ON market_data(label) WHERE label IS NULL;
