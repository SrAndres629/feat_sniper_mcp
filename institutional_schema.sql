-- ============================================================================
-- FEAT NEXUS - Institutional Trading Database Schema
-- ============================================================================
-- Designed for High Frequency Trading (HFT) and ML Pipeline
-- This is a CLEAN schema for a NEW Supabase project
-- DO NOT run this on an existing project with data
-- ============================================================================

-- ============================================================================
-- EXTENSIONS
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- For embeddings (pgvector)

-- ============================================================================
-- ENUMS
-- ============================================================================
DO $$ BEGIN
    CREATE TYPE asset_class AS ENUM ('METAL', 'FOREX', 'CRYPTO', 'INDEX', 'COMMODITY');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE knowledge_scope AS ENUM ('UNIVERSAL', 'ASSET_SPECIFIC', 'CLASS_SPECIFIC');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE trade_direction AS ENUM ('BUY', 'SELL', 'WAIT');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE trade_outcome AS ENUM ('WIN', 'LOSS', 'BREAKEVEN', 'PENDING');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE fsm_state AS ENUM ('CALIBRATING', 'ACCUMULATION', 'MANIPULATION', 'EXPANSION', 'DISTRIBUTION');
EXCEPTION WHEN duplicate_object THEN null;
END $$;

-- ============================================================================
-- TABLE: assets_config
-- Master catalog of tradeable assets
-- ============================================================================
CREATE TABLE IF NOT EXISTS assets_config (
    symbol VARCHAR(20) PRIMARY KEY,
    asset_class asset_class NOT NULL,
    
    -- Contract specifications
    pip_value DECIMAL(10,6) NOT NULL,
    digits INTEGER DEFAULT 5,
    contract_size DECIMAL(15,2) DEFAULT 100000,
    
    -- Volatility parameters
    typical_atr_m5 DECIMAL(15,5) NOT NULL,
    typical_atr_h1 DECIMAL(15,5),
    spread_tolerance_atr DECIMAL(5,3) DEFAULT 0.10,
    
    -- Trading windows (UTC hours as array)
    killzone_hours INTEGER[] DEFAULT ARRAY[7,8,9,13,14,15],  -- London + NY
    avoid_hours INTEGER[] DEFAULT ARRAY[22,23,0,1,2,3],       -- Asian low liquidity
    
    -- Correlations
    correlated_assets TEXT[],
    inverse_correlations TEXT[],
    
    -- Risk limits
    max_position_size DECIMAL(10,2) DEFAULT 1.0,
    max_daily_trades INTEGER DEFAULT 10,
    
    -- Metadata
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Default assets
INSERT INTO assets_config (symbol, asset_class, pip_value, typical_atr_m5, killzone_hours, correlated_assets)
VALUES 
    ('XAUUSD', 'METAL', 0.01, 2.5, ARRAY[7,8,9,13,14,15], ARRAY['XAGUSD']),
    ('EURUSD', 'FOREX', 0.0001, 0.0015, ARRAY[7,8,9,13,14,15], ARRAY['GBPUSD']),
    ('GBPUSD', 'FOREX', 0.0001, 0.0018, ARRAY[7,8,9,13,14,15], ARRAY['EURUSD']),
    ('BTCUSD', 'CRYPTO', 1.0, 150.0, ARRAY[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], ARRAY['ETHUSD']),
    ('US30', 'INDEX', 1.0, 50.0, ARRAY[13,14,15,16,17,18,19,20], ARRAY['SPX500', 'NAS100'])
ON CONFLICT (symbol) DO NOTHING;

-- ============================================================================
-- TABLE: market_ticks
-- High-frequency tick data (optimized for time-series)
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_ticks (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    
    -- Price data
    bid DECIMAL(15,5) NOT NULL,
    ask DECIMAL(15,5) NOT NULL,
    
    -- Volume and flags
    volume DECIMAL(20,2),
    tick_flags INTEGER DEFAULT 0,  -- Bitflags for special conditions
    
    -- Computed on insert
    spread_points DECIMAL(10,2),
    mid_price DECIMAL(15,5),
    
    -- Timestamp (primary index)
    tick_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Partition hint
    partition_date DATE GENERATED ALWAYS AS (tick_time::DATE) STORED
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time ON market_ticks(symbol, tick_time DESC);
CREATE INDEX IF NOT EXISTS idx_ticks_date ON market_ticks(partition_date);

-- ============================================================================
-- TABLE: feat_context
-- FEAT strategy state snapshots
-- ============================================================================
CREATE TABLE IF NOT EXISTS feat_context (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    
    -- FEAT Pillars
    feat_score DECIMAL(5,2),
    
    -- Form
    has_bos BOOLEAN DEFAULT FALSE,
    has_choch BOOLEAN DEFAULT FALSE,
    has_intent_candle BOOLEAN DEFAULT FALSE,
    compression_ratio DECIMAL(5,3),
    
    -- Space  
    at_pvp_zone BOOLEAN DEFAULT FALSE,
    pvp_distance_atr DECIMAL(5,3),
    zone_type VARCHAR(20),
    
    -- Acceleration
    velocity DECIMAL(10,5),
    momentum DECIMAL(10,5),
    rsi DECIMAL(5,2),
    is_exhausted BOOLEAN DEFAULT FALSE,
    
    -- Time
    is_killzone BOOLEAN DEFAULT FALSE,
    active_session VARCHAR(20),
    h4_direction INTEGER,
    
    -- FSM
    fsm_state fsm_state DEFAULT 'CALIBRATING',
    
    -- Current market
    current_price DECIMAL(15,5),
    current_atr DECIMAL(15,5),
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feat_symbol_time ON feat_context(symbol, created_at DESC);

-- ============================================================================
-- TABLE: feat_signals
-- Trading signals generated by the system
-- ============================================================================
CREATE TABLE IF NOT EXISTS feat_signals (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    -- Asset context
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    
    -- Signal
    direction trade_direction NOT NULL,
    confidence DECIMAL(4,3) NOT NULL,  -- 0.000 to 1.000
    
    -- ML Details
    ml_source VARCHAR(20),  -- 'GBM', 'LSTM', 'ENSEMBLE', 'RULES'
    p_win DECIMAL(4,3),
    anomaly_score DECIMAL(4,3),
    is_anomaly BOOLEAN DEFAULT FALSE,
    
    -- FEAT context (denormalized for fast queries)
    feat_score DECIMAL(5,2),
    fsm_state fsm_state,
    
    -- Price at signal
    entry_price DECIMAL(15,5),
    current_atr DECIMAL(15,5),
    
    -- Recommended levels (in ATR units)
    sl_atr_distance DECIMAL(4,2) DEFAULT 1.5,
    tp_atr_distance DECIMAL(4,2) DEFAULT 3.0,
    
    -- Execution
    execution_enabled BOOLEAN DEFAULT FALSE,
    executed BOOLEAN DEFAULT FALSE,
    execution_time TIMESTAMPTZ,
    
    -- Outcome tracking
    outcome trade_outcome DEFAULT 'PENDING',
    pnl_atr DECIMAL(6,3),
    pnl_usd DECIMAL(15,2),
    
    -- Reasoning (for N8N agent)
    top_drivers TEXT[],
    explanation TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON feat_signals(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_outcome ON feat_signals(outcome) WHERE outcome = 'PENDING';

-- ============================================================================
-- TABLE: ml_inference_logs
-- Complete audit trail of ML predictions
-- ============================================================================
CREATE TABLE IF NOT EXISTS ml_inference_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    -- Context
    symbol VARCHAR(20) NOT NULL,
    signal_id UUID REFERENCES feat_signals(id),
    
    -- Input features (ATR-normalized)
    input_features JSONB NOT NULL,
    
    -- Model output
    model_name VARCHAR(30),
    raw_prediction DECIMAL(4,3),
    prediction_class trade_direction,
    
    -- Explainability
    feature_importance JSONB,  -- {"rsi": 0.25, "feat_score": 0.4, ...}
    shap_values JSONB,         -- SHAP explanation if available
    
    -- Anomaly detection
    anomaly_score DECIMAL(4,3),
    anomaly_features TEXT[],
    
    -- Performance
    inference_time_ms INTEGER,
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inference_symbol ON ml_inference_logs(symbol, created_at DESC);

-- ============================================================================
-- TABLE: knowledge_base
-- RAG memory for the trading agent
-- ============================================================================
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    
    -- Classification
    scope knowledge_scope NOT NULL DEFAULT 'UNIVERSAL',
    asset_class asset_class,
    symbol VARCHAR(20),
    
    -- Content
    category VARCHAR(50) NOT NULL,  -- 'FEAT_PATTERN', 'KILLZONE_RULE', 'PVP_BEHAVIOR', etc.
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    
    -- Learning metrics
    confidence DECIMAL(3,2) DEFAULT 0.50,
    times_validated INTEGER DEFAULT 0,
    times_failed INTEGER DEFAULT 0,
    
    -- Vector embedding (pgvector)
    embedding vector(384),
    
    -- Metadata
    source VARCHAR(100),  -- 'USER', 'ORACLE_LEARNING', 'PATTERN_DETECTION'
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_knowledge_scope ON knowledge_base(scope);
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_base(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_symbol ON knowledge_base(symbol);

-- ============================================================================
-- TABLE: bot_activity_log
-- Audit trail of all bot actions
-- ============================================================================
CREATE TABLE IF NOT EXISTS bot_activity_log (
    id BIGSERIAL PRIMARY KEY,
    
    -- Action details
    action_type VARCHAR(50) NOT NULL, -- 'SIGNAL_GENERATED', 'TRADE_EXECUTED', 'MEMORY_STORED', etc.
    action_data JSONB,
    
    -- Context
    symbol VARCHAR(20),
    related_signal_id UUID REFERENCES feat_signals(id),
    
    -- Status
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_activity_time ON bot_activity_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_type ON bot_activity_log(action_type);

-- ============================================================================
-- ROW LEVEL SECURITY
-- ============================================================================
ALTER TABLE assets_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_ticks ENABLE ROW LEVEL SECURITY;
ALTER TABLE feat_context ENABLE ROW LEVEL SECURITY;
ALTER TABLE feat_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_inference_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_activity_log ENABLE ROW LEVEL SECURITY;

-- Service role full access
CREATE POLICY "Service role full access" ON assets_config FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON market_ticks FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON feat_context FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON feat_signals FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON ml_inference_logs FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON knowledge_base FOR ALL USING (auth.role() = 'service_role');
CREATE POLICY "Service role full access" ON bot_activity_log FOR ALL USING (auth.role() = 'service_role');

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Get asset profile with current state
CREATE OR REPLACE FUNCTION get_asset_profile(p_symbol VARCHAR(20))
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'symbol', ac.symbol,
        'asset_class', ac.asset_class,
        'pip_value', ac.pip_value,
        'typical_atr_m5', ac.typical_atr_m5,
        'killzone_hours', ac.killzone_hours,
        'is_killzone_now', (EXTRACT(HOUR FROM NOW() AT TIME ZONE 'UTC')::INTEGER = ANY(ac.killzone_hours)),
        'correlated_assets', ac.correlated_assets
    )
    INTO result
    FROM assets_config ac
    WHERE ac.symbol = p_symbol;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Update learning confidence based on outcomes
CREATE OR REPLACE FUNCTION update_learning_confidence()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.outcome IN ('WIN', 'LOSS') AND OLD.outcome = 'PENDING' THEN
        -- Update related knowledge entries
        UPDATE knowledge_base
        SET 
            times_validated = times_validated + CASE WHEN NEW.outcome = 'WIN' THEN 1 ELSE 0 END,
            times_failed = times_failed + CASE WHEN NEW.outcome = 'LOSS' THEN 1 ELSE 0 END,
            confidence = CASE 
                WHEN (times_validated + times_failed + 1) > 0 
                THEN (times_validated + CASE WHEN NEW.outcome = 'WIN' THEN 1 ELSE 0 END)::DECIMAL / 
                     (times_validated + times_failed + 1)
                ELSE 0.50
            END,
            updated_at = NOW()
        WHERE 
            (scope = 'UNIVERSAL' AND category = NEW.fsm_state::TEXT)
            OR (scope = 'ASSET_SPECIFIC' AND symbol = NEW.symbol);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_learning
    AFTER UPDATE ON feat_signals
    FOR EACH ROW
    WHEN (NEW.outcome IS DISTINCT FROM OLD.outcome)
    EXECUTE FUNCTION update_learning_confidence();

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Daily performance summary
CREATE OR REPLACE VIEW daily_performance AS
SELECT 
    DATE(created_at) as trade_date,
    symbol,
    COUNT(*) as total_signals,
    SUM(CASE WHEN executed THEN 1 ELSE 0 END) as executed,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
    ROUND(AVG(confidence)::DECIMAL, 3) as avg_confidence,
    ROUND(SUM(COALESCE(pnl_atr, 0))::DECIMAL, 2) as total_pnl_atr
FROM feat_signals
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at), symbol
ORDER BY trade_date DESC, symbol;

-- ML model accuracy
CREATE OR REPLACE VIEW ml_accuracy AS
SELECT 
    model_name,
    COUNT(*) as total_predictions,
    SUM(CASE 
        WHEN fs.outcome = 'WIN' AND il.prediction_class IN ('BUY', 'SELL') THEN 1
        WHEN fs.outcome = 'LOSS' AND il.prediction_class = 'WAIT' THEN 1
        ELSE 0
    END) as correct,
    ROUND(AVG(il.raw_prediction)::DECIMAL, 3) as avg_confidence
FROM ml_inference_logs il
LEFT JOIN feat_signals fs ON il.signal_id = fs.id
WHERE fs.outcome IN ('WIN', 'LOSS')
GROUP BY model_name;

-- ============================================================================
-- CONFIRMATION
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE '✅ FEAT NEXUS Institutional Schema installed successfully!';
    RAISE NOTICE 'Tables created: assets_config, market_ticks, feat_context, feat_signals, ml_inference_logs, knowledge_base, bot_activity_log';
END $$;
