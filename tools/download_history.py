
import os
import sys
import asyncio
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.mt5_conn import mt5_conn
from app.ml.data_collector import data_collector, OracleLabeler
from app.core.config import settings
import MetaTrader5 as mt5

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FEAT.DataEngineer")

async def download_history_bulk(symbol: str, count: int = 50000):
    """
    [DATA ENGINEER PROTOCOL]
    Downloads high-fidelity historical data from MT5 and populates the SQLite memory.
    """
    # 1. MT5 Startup
    success = await mt5_conn.startup()
    if not success:
        logger.error("❌ Failed to initialize MT5 Connection.")
        return

    timeframes = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4
    }

    total_inserted = 0

    for tf_name, tf_mt5 in timeframes.items():
        logger.info(f"📥 Downloading {count} candles for {tf_name}...")
        
        # Download from MT5
        # rates: [time, open, high, low, close, tick_volume, spread, real_volume]
        rates = await mt5_conn.execute(mt5.copy_rates_from_pos, symbol, tf_mt5, 0, count)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"⚠️ No data received for {tf_name}")
            continue

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['tick_time'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Map MT5 columns to DB columns
        records = []
        for _, row in df.iterrows():
            records.append({
                "tick_time": row['tick_time'],
                "symbol": symbol,
                "timeframe": tf_name,
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['tick_volume'])
            })

        # Insert into DB
        with data_collector.db.get_connection() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO market_data (
                    tick_time, symbol, timeframe, open, high, low, close, volume
                ) VALUES (
                    :tick_time, :symbol, :timeframe, :open, :high, :low, :close, :volume
                )
            """, records)
            conn.commit()
            
        inserted = len(records)
        total_inserted += inserted
        logger.info(f"✅ Inserted {inserted} samples for {tf_name}.")

    # 2. Oracle Labeling (Lookahead 10 candles by default)
    logger.info("🧠 Running Oracle Labeler to generate training targets...")
    oracle = OracleLabeler(data_collector.db, lookahead=10, threshold=0.001) # 0.1% Profit Threshold
    
    for tf_name in timeframes.keys():
        labeled = oracle.process_pending_labels(symbol, tf_name)
        logger.info(f"🏷️ Labeled {labeled} samples for {tf_name}.")

    logger.info(f"📊 SUMMARY: {total_inserted} total records prepared.")
    print("\n✅ DATOS LISTOS. EJECUTA EL ENTRENAMIENTO.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="XAUUSD")
    parser.add_argument("--count", type=int, default=50000)
    args = parser.parse_args()

    asyncio.run(download_history_bulk(args.symbol, args.count))
