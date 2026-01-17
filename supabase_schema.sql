-- FEAT SNIPER CORE SCHEMA

-- 1. Table for Real-time Signals
CREATE TABLE IF NOT EXISTS public.sniper_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    action TEXT NOT NULL, -- BUY, SELL, WAIT
    price DECIMAL NOT NULL,
    confidence DECIMAL,
    engineer_diagnosis TEXT,
    tactician_poi TEXT,
    sl DECIMAL,
    tp DECIMAL,
    metadata JSONB -- Extra ML metrics (deltaFlow, RSI, etc)
);

-- 1b. Table for High-Frequency Neural State (Dashboard Stream)
CREATE TABLE IF NOT EXISTS public.neural_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    symbol TEXT NOT NULL,
    price DECIMAL NOT NULL,
    alpha_confidence DECIMAL,
    acceleration DECIMAL,
    hurst DECIMAL,
    
    -- PVP Context (Level 46)
    poc_price DECIMAL,
    vah_price DECIMAL,
    val_price DECIMAL,
    energy_score DECIMAL,
    
    metadata JSONB
);
ALTER PUBLICATION supabase_realtime ADD TABLE neural_signals;

-- 2. Table for Model Performance Tracking
CREATE TABLE IF NOT EXISTS public.model_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    win_rate DECIMAL,
    profit_factor DECIMAL,
    sharpe_ratio DECIMAL,
    last_training TIMESTAMP WITH TIME ZONE,
    hyperparameters JSONB
);

-- 3. Table for Trade Execution (Sync with MT5)
CREATE TABLE IF NOT EXISTS public.sniper_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mt5_ticket BIGINT UNIQUE,
    symbol TEXT NOT NULL,
    type TEXT NOT NULL,
    volume DECIMAL NOT NULL,
    entry_price DECIMAL NOT NULL,
    exit_price DECIMAL,
    profit DECIMAL,
    status TEXT DEFAULT 'OPEN', -- OPEN, CLOSED, CANCELLED
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    closed_at TIMESTAMP WITH TIME ZONE
);

-- Enable Realtime for signals
ALTER PUBLICATION supabase_realtime ADD TABLE sniper_signals;
