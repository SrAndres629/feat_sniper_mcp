-- FEAT NEXUS: INSTITUTIONAL TELEMETRY SCHEMA
-- This aligns with dashboard.html and app/core/streamer.py

-- 1. Logs Table
CREATE TABLE IF NOT EXISTS public.bot_activity_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    module TEXT,
    message TEXT,
    session_id TEXT,
    level TEXT -- Optional but good for filtering
);

-- 2. Live Metrics Table (State)
CREATE TABLE IF NOT EXISTS public.live_metrics (
    session_id TEXT PRIMARY KEY,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    balance DECIMAL,
    equity DECIMAL,
    margin_free DECIMAL,
    pnl_daily DECIMAL
);

-- 3. Neural Signals Table (High Fidelity)
CREATE TABLE IF NOT EXISTS public.neural_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    session_id TEXT,
    alpha_confidence DECIMAL,
    acceleration DECIMAL,
    hurst DECIMAL,
    price DECIMAL
);

-- 4. Enable Realtime
ALTER PUBLICATION supabase_realtime ADD TABLE bot_activity_log;
ALTER PUBLICATION supabase_realtime ADD TABLE live_metrics;
ALTER PUBLICATION supabase_realtime ADD TABLE neural_signals;
