import os
import asyncio
import pandas as pd
import logging
from datetime import datetime
from app.core.mt5_conn import mt5_conn, mt5

# Logging Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NexusTraining.Downloader")

async def download_historical_data(symbol: str, timeframe: str, n_bars: int = 50000):
    """
    Downloads historical OHLCV data from MT5 for AI training.
    """
    try:
        logger.info(f"Connecting to MT5 to download {n_bars} bars for {symbol} ({timeframe})...")
        await mt5_conn.connect()
        
        # Check if symbol is available
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found.")
            return

        # Select symbol
        await mt5_conn.execute(mt5.symbol_select, symbol, True)
        
        # Timeframe mapping
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "H1": mt5.TIMEFRAME_H1
        }
        mt5_tf = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_M1)

        # Download rates
        logger.info("Requesting rates from MT5...")
        rates = await mt5_conn.execute(mt5.copy_rates_from_pos, symbol, mt5_tf, 0, n_bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to download rates. MT5 Error: {await mt5_conn.execute(mt5.last_error)}")
            return

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Save to raw data directory
        raw_dir = "data/raw"
        os.makedirs(raw_dir, exist_ok=True)
        
        filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = os.path.join(raw_dir, filename)
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Successfully downloaded {len(df)} bars.")
        logger.info(f"Data saved to: {filepath}")
        
        return filepath

    except Exception as e:
        logger.error(f"Error during download: {e}")
    finally:
        await mt5_conn.shutdown()

if __name__ == "__main__":
    # Settings for the download
    SYMBOL = "XAUUSD" # Gold is the standard for Feat Sniper
    TIMEFRAME = "M1"
    COUNT = 55000 # 50k + 5k buffer for indicators/lag
    
    asyncio.run(download_historical_data(SYMBOL, TIMEFRAME, COUNT))
