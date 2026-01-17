
import asyncio
import logging
import MetaTrader5 as mt5
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from app.core.mt5_conn import mt5_conn
from app.ml.ml_engine import ml_engine
from app.services.auto_tuner import auto_tuner
from app.ml.feat_processor import feat_processor
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FEAT.Evolution")

async def run_evolution_cycle(symbol: str = "XAUUSD", days: int = 180):
    """
    [LEVEL 34] ORCHESTRATED MASS EVOLUTION
    1. Synchronize MT5 History (180 Days)
    2. Tensorize History -> Experience with RLAIF Context
    3. Run Genetic Optimization
    """
    logger.info(f"📊 STARTING MASS EVOLUTION CYCLE FOR {symbol} ({days} days)")
    
    if not mt5_conn.connected:
        await mt5_conn.startup()

    # 1. Fetch History
    # 180 days of M1 is roughly 180 * 1440 = 259,200 bars.
    num_bars = days * 1440
    logger.info(f"📥 Downloading {num_bars} M1 bars for {symbol}...")
    
    # Ensure symbol is visible in Market Watch
    mt5.symbol_select(symbol, True)
    
    MAX_BARS_PER_REQUEST = 90000
    dfs = []
    
    if num_bars <= MAX_BARS_PER_REQUEST:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, num_bars)
        if rates is not None and len(rates) > 0:
            dfs.append(pd.DataFrame(rates))
    else:
        logger.info(f"🔄 Requesting {num_bars} bars in chunks...")
        for start in range(0, num_bars, MAX_BARS_PER_REQUEST):
            count = min(MAX_BARS_PER_REQUEST, num_bars - start)
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, start, count)
            if rates is not None and len(rates) > 0:
                dfs.append(pd.DataFrame(rates))
            else:
                logger.warning(f"⚠️ Could not fetch chunk starting at {start}")
                break
    
    if not dfs:
        err = mt5.last_error()
        logger.error(f"❌ Failed to fetch history for {symbol}. Error: {err}")
        return

    df = pd.concat(dfs).drop_duplicates(subset=['time']).sort_values('time')
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 2. Replay & Generate Experience Memory
    memory_file = "data/experience_memory.jsonl"
    os.makedirs("data", exist_ok=True)
    
    # Clear old memory if mass training
    if days > 30 and os.path.exists(memory_file):
        logger.info("🧹 Clearing legacy experience for fresh mass training...")
        open(memory_file, 'w').close()
    
    logger.info(f"🧠 Processing {len(df)} candles into Experience Memory...")
    
    entries = []
    # Sample every 30 minutes to generate diverse trade experiences
    for i in range(100, len(df) - 60, 30):
        current = df.iloc[i]
        
        # [RLAIF CONTEXT SIMULATION]
        # In a real replay, we'd run the actual FEAT Indicator code.
        # Here we simulate valid/invalid structure based on price movement
        price_std = df['close'].iloc[i-50:i].std()
        
        # Simulating FEAT Score: High if price is moving with momentum
        momentum = abs(current['close'] - df.iloc[i-10]['close'])
        feat_score = 40 + (min(momentum / (price_std + 1e-6), 1.0) * 50)
        acceleration = (current['close'] - df.iloc[i-1]['close']) / (price_std + 1e-6)
        
        # Simulated prediction
        confidence = 0.5 + (random.random() * 0.4)
        
        # Outcome (Look ahead 30 mins)
        future_idx = i + 30
        future_price = df.iloc[future_idx]['close']
        pnl_pts = (future_price - current['close']) / 0.01 # Pts for Gold
        
        record = {
            "timestamp": current['time'].isoformat(),
            "symbol": symbol,
            "state_snapshot": {
                "close": current['close'],
                "confidence": confidence,
                "feat_score": round(feat_score, 2),
                "acceleration": round(acceleration, 4),
                "symbol": symbol
            },
            "pnl": round(pnl_pts, 2),
            "action": "BUY" if pnl_pts > 0 else "SELL"
        }
        entries.append(json.dumps(record) + "\n")

    with open(memory_file, "a") as f:
        f.writelines(entries)

    # 3. Trigger GA
    logger.info(f"🧬 Invoking Genetic Algorithm on {len(entries)} virtual trades...")
    # Increase generations for mass training
    auto_tuner.generations = 25 
    auto_tuner.evolve_population()
    
    logger.info("✅ MASS EVOLUTION COMPLETE. Check Dashboard 'Neural Evolution' tab.")

if __name__ == "__main__":
    import random
    import sys
    
    # Check for CLI args override
    days_arg = 180
    if len(sys.argv) > 1:
        try: days_arg = int(sys.argv[1])
        except: pass
        
    asyncio.run(run_evolution_cycle(days=days_arg))
