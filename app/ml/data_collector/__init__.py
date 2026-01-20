from .engine import DataCollector
from .constants import DB_PATH, TIMEFRAMES, TIMEFRAME_MAP, SystemState
from .cvd import fetch_tick_data, compute_real_cvd, fetch_historical_ticks
from .labeler import OracleLabeler

data_collector = DataCollector()

async def collect_sample(symbol, candle, indicators):
    data_collector.collect(symbol, candle, indicators)
    return data_collector.get_stats()

async def collect_ticks(symbol, tick_record):
    data_collector.collect_ticks(symbol, tick_record)
    return data_collector.get_stats()

async def get_collection_stats():
    return data_collector.get_stats()

async def flush_pending():
    data_collector.force_flush()
    return {"status": "flushed", **data_collector.get_stats()}

__all__ = [
    "data_collector", "collect_sample", "collect_ticks", "get_collection_stats", "flush_pending",
    "fetch_tick_data", "compute_real_cvd", "fetch_historical_ticks",
    "OracleLabeler",
    "SystemState", "TIMEFRAMES", "TIMEFRAME_MAP"
]
