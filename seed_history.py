import MetaTrader5 as mt5
import pandas as pd
import sqlite3
import numpy as np
import os
import logging
from datetime import datetime, timedelta

# Configuration
SYMBOL = "BTCUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
BARS = 10000  # ~7 days of M1 data
DB_PATH = "data/market_data.db"
LOG_LEVEL = "INFO"

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger("SeedHistory")

def main():
    logger.info(">>> connecting to MetaTrader 5...")
    if not mt5.initialize():
        logger.error(f"initialize() failed, error code = {mt5.last_error()}")
        return

    logger.info(f">>> Downloading {BARS} bars for {SYMBOL}...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        logger.error("No data received!")
        return

    df = pd.DataFrame(rates)
    logger.info(f"Columns received: {df.columns.tolist()}")

    # Rename tick_volume to volume for consistency
    if 'tick_volume' in df.columns:
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    elif 'real_volume' in df.columns:
        df.rename(columns={'real_volume': 'volume'}, inplace=True)
    else:
        logger.warning("No volume column found! Creating dummy volume.")
        df['volume'] = 1.0

    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Feature Engineering (Simplified)
    logger.info(">>> Computing features...")
    
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float) # Ensure float
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_spread'] = df['ema_fast'] - df['ema_slow']
    df['feat_score'] = 0.5  # Placeholder
    df['fsm_state'] = 0     # Placeholder
    df['atr'] = calculate_atr(df, 14)
    df['compression'] = 0.5 # Placeholder
    df['liquidity_above'] = 0.0
    df['liquidity_below'] = 0.0
    
    # Labeling (Simple "Next Close Return")
    df['next_close'] = df['close'].shift(-1)
    df['return'] = (df['next_close'] - df['close']) / df['close']
    df['label'] = (df['return'] > 0).astype(int) # 1 if price goes up, 0 otherwise
    
    # CLEAN DATA: Replace Inf with NaN and drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Insert into SQLite
    logger.info(f">>> Inserting {len(df)} records into {DB_PATH}...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Update Ticks Table
    # Create tables if not exist (using simpler schema for seeding)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick_time TEXT, symbol TEXT, close REAL, open REAL, high REAL, low REAL, volume REAL,
            rsi REAL, ema_fast REAL, ema_slow REAL, ema_spread REAL,
            feat_score REAL, fsm_state REAL, atr REAL, compression REAL,
            liquidity_above REAL, liquidity_below REAL,
            label INTEGER DEFAULT NULL, labeled_at TEXT DEFAULT NULL
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick_id INTEGER, tick_time TEXT, symbol TEXT,
            close REAL, open REAL, high REAL, low REAL, volume REAL,
            rsi REAL, ema_fast REAL, ema_slow REAL, ema_spread REAL,
            feat_score REAL, fsm_state REAL, atr REAL, compression REAL,
            liquidity_above REAL, liquidity_below REAL,
            label INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Convert to list of tuples for execuymany
    rows = []
    for _, row in df.iterrows():
        rows.append((
            row['time'].isoformat(), SYMBOL, row['close'], row['open'], row['high'], row['low'], row['volume'],
            row['rsi'], row['ema_fast'], row['ema_slow'], row['ema_spread'],
            row['feat_score'], row['fsm_state'], row['atr'], row['compression'],
            row['liquidity_above'], row['liquidity_below'], int(row['label'])
        ))

    # Insert into training_samples (Ready for training)
    conn.executemany('''
        INSERT INTO training_samples (
            tick_time, symbol, close, open, high, low, volume,
            rsi, ema_fast, ema_slow, ema_spread,
            feat_score, fsm_state, atr, compression,
            liquidity_above, liquidity_below, label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', rows)
    
    conn.commit()
    conn.close()
    
    # Export CSV for train_models.py (Legacy support)
    csv_path = "data/training_dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f">>> CSV exported to {csv_path}")
    logger.info(">>> DONE. You can now run 'python app/ml/train_models.py'")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

if __name__ == "__main__":
    main()
