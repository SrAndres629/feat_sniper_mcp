
import MetaTrader5 as mt5
import pandas as pd
import sqlite3
import numpy as np
import os
import logging
from datetime import datetime, timedelta, timezone
from app.core.config import settings
from app.ml.data_collector import data_collector

# Configuration
SYMBOL = settings.SYMBOL
TIMEFRAME = mt5.TIMEFRAME_M1
BARS_INITIAL = 10000 
DB_PATH = "data/market_data.db"
LOG_LEVEL = "INFO"

logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(f"SeedHistory_{SYMBOL}")

def get_last_timestamp(conn, symbol):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(tick_time) FROM market_data WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        if result and result[0]:
            return datetime.fromisoformat(result[0].replace("Z", "+00:00"))
    except Exception as e:
        logger.warning(f"Could not fetch last timestamp: {e}")
    return None

def main():
    logger.info(f">>> connecting to MetaTrader 5 for {SYMBOL}...")
    if not mt5.initialize():
        logger.error(f"initialize() failed, error code = {mt5.last_error()}")
        return

    # 1. Ensure Schema
    logger.info(">>> Ensuring database schema (via DataCollector)...")
    with data_collector.db.get_connection() as conn:
        last_ts = get_last_timestamp(conn, SYMBOL)
    
        rates = None
        if last_ts:
            logger.info(f" Existing data found for {SYMBOL} up to {last_ts}.")
            date_from = last_ts
            date_to = datetime.now(timezone.utc)
            rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, date_from, date_to)
        else:
            logger.info(f" No data for {SYMBOL}. Starting GENESIS download ({BARS_INITIAL} bars)...")
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_INITIAL)
            
        mt5.shutdown()

        if rates is None or len(rates) == 0:
            logger.info(f"No new data received for {SYMBOL}.")
            return

        df = pd.DataFrame(rates)
        if 'tick_volume' in df.columns:
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        logger.info(">>> Global Feature Engineering (Cold Start)...")
        df['close'] = df['close'].astype(float)
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['atr'] = calculate_atr(df, 14)
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # neutrals for new features
        neutrals = {
            'feat_score': 50.0, 'fsm_state': 0, 'liquidity_ratio': 1.0, 'volatility_zscore': 0.0,
            'momentum_kinetic_micro': 0.0, 'entropy_coefficient': 0.5, 'cycle_harmonic_phase': 0.0,
            'institutional_mass_flow': 0.0, 'volatility_regime_norm': 0.0, 'acceptance_ratio': 1.0,
            'wick_stress': 0.0, 'poc_z_score': 0.0, 'cvd_acceleration': 0.0,
            'micro_comp': 0.5, 'micro_slope': 0.0, 'oper_slope': 0.0, 'macro_slope': 0.0,
            'bias_slope': 0.0, 'fan_bullish': 0.0
        }
        for k, v in neutrals.items():
            df[k] = v
        
        # Labeling
        df['next_close'] = df['close'].shift(-1)
        df['return'] = (df['next_close'] - df['close']) / df['close']
        df['label'] = (df['return'] > 0).astype(int) 
        
        df.dropna(inplace=True)

        logger.info(f">>> Inserting {len(df)} records for {SYMBOL}...")
        
        columns = [
            "tick_time", "symbol", "timeframe", "close", "open", "high", "low", "volume",
            "rsi", "atr", "ema_fast", "ema_slow", "feat_score", "fsm_state",
            "liquidity_ratio", "volatility_zscore", "momentum_kinetic_micro",
            "entropy_coefficient", "cycle_harmonic_phase", "institutional_mass_flow",
            "volatility_regime_norm", "acceptance_ratio", "wick_stress", "poc_z_score",
            "cvd_acceleration", "micro_comp", "micro_slope", "oper_slope",
            "macro_slope", "bias_slope", "fan_bullish", "label"
        ]
        
        query = f"INSERT OR IGNORE INTO market_data ({', '.join(columns)}) VALUES ({', '.join(['?']*len(columns))})"
        
        data_to_insert = []
        for _, row in df.iterrows():
            data_to_insert.append((
                row['time'].isoformat(), SYMBOL, "M1", row['close'], row['open'], row['high'], row['low'], row['volume'],
                row['rsi'], row['atr'], row['ema_fast'], row['ema_slow'], row['feat_score'], row['fsm_state'],
                row['liquidity_ratio'], row['volatility_zscore'], row['momentum_kinetic_micro'],
                row['entropy_coefficient'], row['cycle_harmonic_phase'], row['institutional_mass_flow'],
                row['volatility_regime_norm'], row['acceptance_ratio'], row['wick_stress'], row['poc_z_score'],
                row['cvd_acceleration'], row['micro_comp'], row['micro_slope'], row['oper_slope'],
                row['macro_slope'], row['bias_slope'], row['fan_bullish'], int(row['label'])
            ))
            
        conn.executemany(query, data_to_insert)
        conn.commit()
    
    logger.info(">>> GENESIS SEEDING COMPLETE.")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high = df['high']; low = df['low']; close = df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

if __name__ == "__main__":
    main()
