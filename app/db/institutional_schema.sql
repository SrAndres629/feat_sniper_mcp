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
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    -- Technical Tensors
    rsi REAL,
    atr REAL,
    ema_fast REAL,
    ema_slow REAL,
    fsm_state INTEGER,
    feat_score REAL,
    -- Contextual Data
    liquidity_ratio REAL,
    volatility_zscore REAL,
    -- Neural Tensors
    momentum_kinetic_micro REAL,
    entropy_coefficient REAL,
    cycle_harmonic_phase REAL,
    institutional_mass_flow REAL,
    volatility_regime_norm REAL,
    acceptance_ratio REAL,
    wick_stress REAL,
    -- PVP / CVD Tensors
    poc_z_score REAL,
    cvd_acceleration REAL,
    -- Ribbon Physics (Multifractal Layers)
    micro_comp REAL,
    micro_slope REAL,
    oper_slope REAL,
    macro_slope REAL,
    bias_slope REAL,
    fan_bullish REAL,
    -- ML Meta-info
    label INTEGER DEFAULT NULL,
    labeled_at TIMESTAMP DEFAULT NULL,
    UNIQUE (tick_time, symbol, timeframe)
);

-- 2. FRACTAL ANALYSIS REGISTRY
CREATE TABLE IF NOT EXISTS fractal_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    hurst_exponent REAL,
    fractal_dimension REAL,
    regime_mode TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. INSTITUTIONAL ACTIVITY LOG
CREATE TABLE IF NOT EXISTS bot_activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event TEXT NOT NULL,
    level TEXT DEFAULT 'INFO',
    phase TEXT,
    details JSON,
    trace_id TEXT
);

-- 4. PERFORMANCE TRACKING
CREATE TABLE IF NOT EXISTS model_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe_context TEXT,
    win_rate REAL,
    profit_factor REAL,
    sharpe_ratio REAL,
    drawdown_max REAL,
    hyperparameters JSON
);

-- INDEXING
CREATE INDEX IF NOT EXISTS idx_market_lookup ON market_data (symbol, timeframe, tick_time DESC);

CREATE INDEX IF NOT EXISTS idx_fractal_lookup ON fractal_analysis (symbol, timeframe, analysis_time DESC);

CREATE INDEX IF NOT EXISTS idx_unlabeled_context ON market_data (label)
WHERE
    label IS NULL;

-- 5. TRAINING VIEW (Labeled Data Only)
CREATE VIEW IF NOT EXISTS training_samples AS
SELECT
    id,
    tick_time as timestamp,
    symbol,
    timeframe,
    close,
    open,
    high,
    low,
    volume,
    -- Technicals
    rsi,
    atr,
    ema_fast,
    ema_slow,
    (ema_fast - ema_slow) as ema_spread,
    -- Neural Features
    feat_score,
    fsm_state,
    liquidity_ratio,
    volatility_zscore,
    momentum_kinetic_micro,
    entropy_coefficient,
    cycle_harmonic_phase,
    institutional_mass_flow,
    volatility_regime_norm,
    acceptance_ratio,
    wick_stress,
    poc_z_score,
    cvd_acceleration,
    -- Physics
    micro_comp,
    micro_slope,
    oper_slope,
    macro_slope,
    bias_slope,
    fan_bullish,
    -- Target
    label
FROM
    market_data
WHERE
    label IS NOT NULL;

-- 6. TICK-LEVEL MICROSTRUCTURE (For high-frequency training)
CREATE TABLE IF NOT EXISTS tick_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_time TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    bid REAL,
    ask REAL,
    bid_vol REAL,
    ask_vol REAL,
    ofi REAL,
    entropy REAL,
    hurst REAL,
    cvd REAL
);

CREATE INDEX IF NOT EXISTS idx_tick_lookup ON tick_data (symbol, tick_time DESC);