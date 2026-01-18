import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger("DataCollector.CVD")

async def fetch_historical_ticks(symbol: str, date_from: datetime, date_to: datetime) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
    except ImportError: return pd.DataFrame()
    if not mt5.terminal_info():
        if not mt5.initialize(): return pd.DataFrame()
    ticks = mt5.copy_ticks_range(symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0: return pd.DataFrame()
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['is_buy'] = (df['flags'] & 32) > 0
    df['is_sell'] = (df['flags'] & 64) > 0
    return df

def compute_real_cvd(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty: return {"cvd": 0.0, "imbalance_ratio": 0.0, "acceleration": 0.0}
    df = df.copy()
    df['signed_volume'] = np.where(df['is_buy'], df['volume'], np.where(df['is_sell'], -df['volume'], 0))
    df['cvd'] = df['signed_volume'].cumsum()
    bv, sv = df.loc[df['is_buy'], 'volume'].sum(), df.loc[df['is_sell'], 'volume'].sum()
    imbalance = (bv - sv) / (bv + sv + 1e-9)
    cvd_s = df['cvd'].values
    acc = (cvd_s[-10:].mean() - cvd_s[-20:-10].mean()) / (abs(cvd_s[-20:-10].mean()) + 1e-9) if len(cvd_s) >= 20 else 0.0
    return {"cvd": float(cvd_s[-1]), "buy_volume": float(bv), "sell_volume": float(sv), "imbalance_ratio": float(imbalance), "acceleration": float(acc)}

async def fetch_tick_data(symbol: str, minutes_back: int = 5) -> Dict[str, Any]:
    dt = datetime.now(timezone.utc)
    df = await fetch_historical_ticks(symbol, dt - timedelta(minutes=minutes_back), dt)
    if df.empty: return {"symbol": symbol, "status": "no_data"}
    return {"symbol": symbol, "status": "success", "ticks_count": len(df), "cvd_metrics": compute_real_cvd(df)}
