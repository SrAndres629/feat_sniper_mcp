import logging
import pandas as pd
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from typing import Optional
import asyncio

from app.core.mt5_conn.manager import MT5Connection
from app.services.supabase_sync import supabase_sync

logger = logging.getLogger("nexus.data_collector")

async def smart_sync_data(symbol: str, timeframe=mt5.TIMEFRAME_M5) -> Optional[pd.DataFrame]:
    """
    [DOCTORAL DATA ARCHITECT]
    Delta-Sync Logic: Fetches only the missing candles to fill Informational Gaps.
    Ensures LSTM context continuity.
    """
    conn = MT5Connection()
    if not conn.connected:
        if not await conn.startup():
            logger.error("❌ Failed to connect to MT5 for data sync.")
            return None

    # 1. GET LAST KNOWN STATE
    logger.info(f"🔄 Checking last known state for {symbol}...")
    last_candle_time_str = await supabase_sync.get_last_candle_timestamp(symbol)
    
    if last_candle_time_str is None:
        # 2a. INITIALIZATION: Fetch default large batch (Initial Seed)
        logger.warning(f"⚠️ No history found for {symbol}. Initializing with full load (50,000 bars)...")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 50000)
        if rates is None or len(rates) == 0:
            logger.error("❌ Failed to fetch initial data seed.")
            return None
        new_data = pd.DataFrame(rates)
        new_data['time'] = pd.to_datetime(new_data['time'], unit='s')
    else:
        # 2b. DELTA SYNC: Fetch range from last_known_time to NOW
        last_candle_time = datetime.fromisoformat(last_candle_time_str.replace('Z', '+00:00'))
        # Offset by 1 second to avoid duplicating the last candle if needed, 
        # but upsert handles overlapping nicely.
        start_time = last_candle_time
        end_time = datetime.now()
        
        logger.info(f"➕ Fetching missing data from {start_time} to {end_time}...")
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        
        if rates is None or len(rates) == 0:
            logger.info(f"✅ Data for {symbol} is already up to date.")
            return None
            
        new_data = pd.DataFrame(rates)
        new_data['time'] = pd.to_datetime(new_data['time'], unit='s')
        
    # 3. SAVE TO DB: Close the gap permanently
    logger.info(f"📥 Downloaded {len(new_data)} new candles. Saving to Supabase...")
    await supabase_sync.save_market_history(symbol, new_data)
    
    return new_data

if __name__ == "__main__":
    # Test execution
    async def test():
        df = await smart_sync_data("XAUUSD")
        if df is not None:
            print(f"Synced {len(df)} candles.")
        else:
            print("No new candles or error.")
            
    asyncio.run(test())
