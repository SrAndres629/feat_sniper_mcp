import logging
import time
from typing import Dict, List, Optional, Any
import pandas as pd

# FAIL-FAST: Use centralized MT5 from mt5_conn (no silent mocks)
from app.core.mt5_conn import mt5_conn, mt5, MT5_AVAILABLE

if not MT5_AVAILABLE:
    raise ImportError(
        "MetaTrader5 library not found. This module requires a real MT5 connection. "
        "Install: pip install MetaTrader5 (Windows only)"
    )

from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.Skills.Market")

# Mapeo de Timeframes
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# Sistema de Cach en Memoria
# Sistema de Cach en Memoria
_candles_cache: Dict[str, Dict[str, Any]] = {}
_account_cache: Dict[str, Any] = {"data": None, "timestamp": 0}
MAX_CACHE_SIZE = 100  # Prevent memory leaks by limiting cache keys

def _clean_cache():
    """Mantiene el tamao del cach bajo control."""
    if len(_candles_cache) > MAX_CACHE_SIZE:
        # Eliminar el 20% ms antiguo
        keys_to_remove = sorted(_candles_cache.keys(), key=lambda k: _candles_cache[k]['timestamp'])[:int(MAX_CACHE_SIZE * 0.2)]
        for k in keys_to_remove:
            del _candles_cache[k]


async def get_candles(symbol: str, timeframe: str, n_candles: int = 100, output_format: str = "json") -> Dict[str, Any]:
    """
    Obtiene las ltimas N velas con sistema de cach de 3 segundos.
    Soporta formato CSV para ahorrar tokens en la ventana de contexto de la IA.
    """
    now = time.time()
    tf_upper = timeframe.upper()
    cache_key = f"{symbol}_{tf_upper}_{n_candles}_{output_format}"
    
    # Intentar obtener de cach
    cached = _candles_cache.get(cache_key)
    if cached and (now - cached["timestamp"]) < settings.MARKET_DATA_CACHE_TTL:
        logger.debug(f"Cach HIT para {cache_key}")
        return cached["data"]
    
    # Validar timeframe
    if tf_upper not in TIMEFRAME_MAP:
        raise ValueError(f"Timeframe {timeframe} no es vlido.")
    
    mt5_tf = TIMEFRAME_MAP[tf_upper]
    
    # Asegurar que el smbolo es visible
    await mt5_conn.execute(mt5.symbol_select, symbol, True)
    
    # Obtener datos de MT5 de forma no bloqueante
    rates = await mt5_conn.execute(mt5.copy_rates_from_pos, symbol, mt5_tf, 0, n_candles)
    
    if rates is None or len(rates) == 0:
        return {
            "status": "error",
            "message": "No se pudieron obtener datos del mercado.",
            "mt5_error": await mt5_conn.execute(mt5.last_error)
        }
    
    # Procesar con Pandas
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s').astype(str)
    
    if output_format == "csv":
        # Formato ultra-compacto: time,o,h,l,c,v
        csv_data = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].to_csv(index=False, header=True)
        data = {
            "status": "success",
            "symbol": symbol,
            "timeframe": tf_upper,
            "format": "csv",
            "candles_csv": csv_data,
            "timestamp": now
        }
    else:
        data = {
            "status": "success",
            "symbol": symbol,
            "timeframe": tf_upper,
            "format": "json",
            "candles": df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].to_dict('records'),
            "timestamp": now
        }
    
    # Guardar en cach con limpieza previa
    if len(_candles_cache) > MAX_CACHE_SIZE:
        _clean_cache()
        
    _candles_cache[cache_key] = {"data": data, "timestamp": now}
    logger.debug(f"Cach MISS para {cache_key}. Datos actualizados.")
    
    return data

async def get_account_metrics() -> Dict[str, Any]:
    """
    Obtiene mtricas de la cuenta con cach de 0.5 segundos.
    """
    now = time.time()
    
    # Intentar obtener de cach
    if _account_cache["data"] and (now - _account_cache["timestamp"]) < settings.ACCOUNT_CACHE_TTL:
        return _account_cache["data"]
    
    # Obtener info de cuenta
    account_info = await mt5_conn.execute(mt5.account_info)
    if account_info is None:
        return {
            "status": "error",
            "message": "Fallo al leer informacin de la cuenta.",
            "mt5_error": await mt5_conn.execute(mt5.last_error)
        }
    
    total_positions = await mt5_conn.execute(mt5.positions_total)
    
    data = {
        "status": "success",
        "balance": account_info.balance,
        "equity": account_info.equity,
        "margin_free": account_info.margin_free,
        "margin_level": (account_info.equity / account_info.margin * 100) if account_info.margin > 0 else 0,
        "positions_total": total_positions,
        "profit": account_info.profit,
        "server": account_info.server,
        "currency": account_info.currency,
        "timestamp": now
    }
    
    # Guardar en cach
    _account_cache["data"] = data
    _account_cache["timestamp"] = now
    
    return data

async def get_volatility_metrics(symbol: str, timeframe: str = "H1", period: int = 14) -> Dict[str, Any]:
    """
    Calcula mtricas de volatilidad (ATR y Spread) para un smbolo.
    """
    tf_upper = timeframe.upper()
    if tf_upper not in TIMEFRAME_MAP:
        raise ValueError(f"Timeframe {timeframe} no es vlido.")
    
    mt5_tf = TIMEFRAME_MAP[tf_upper]
    
    # Obtener velas para calcular ATR (period + 1 para tener suficientes diferencias)
    rates = await mt5_conn.execute(mt5.copy_rates_from_pos, symbol, mt5_tf, 0, period + 1)
    
    if rates is None or len(rates) < period:
        return {"status": "error", "message": "No hay suficientes datos para calcular volatilidad."}
    
    df = pd.DataFrame(rates)
    
    # Clculo correcto de ATR (True Range)
    df['prev_close'] = df['close'].shift(1)
    df['tr0'] = df['high'] - df['low']
    df['tr1'] = (df['high'] - df['prev_close']).abs()
    df['tr2'] = (df['low'] - df['prev_close']).abs()
    
    # El True Range es el mximo de las 3 medidas
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # ATR es la media mvil del TR
    atr = df['tr'].tail(period).mean()
    
    # Obtener Spread actual
    symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    
    if not symbol_info or not tick:
        return {"status": "error", "message": "No se pudo obtener informacin del smbolo para spread."}
    
    spread_points = symbol_info.spread
    spread_value = tick.ask - tick.bid
    
    # Status de volatilidad robusto (Percentil)
    # Comparamos el ATR actual con la media del ATR de los ltimos periodos si es posible, 
    # o usamos la desviacin estndar del TR.
    tr_mean = df['tr'].mean()
    tr_std = df['tr'].std()
    
    current_tr = df['tr'].iloc[-1]
    
    if current_tr > (tr_mean + 2 * tr_std):
        status = "EXTREME"
    elif current_tr > (tr_mean + tr_std):
        status = "HIGH"
    elif current_tr < (tr_mean - 0.5 * tr_std):
        status = "LOW"
    else:
        status = "NORMAL"
    
    return {
        "status": "success",
        "symbol": symbol,
        "atr": float(atr),
        "spread": float(spread_value),
        "spread_points": int(spread_points),
        "volatility_status": status
    }

async def get_snapshot(symbol: str, timeframe: str = "M5") -> Dict[str, Any]:
    """
    Obtiene una radiografa completa del mercado en un solo llamado.
    Combina: ltima vela, Quote actual, Mtricas de Cuenta y Volatilidad.
    Esencial para que el agente tome decisiones rpidas (Ojo de Halcn).
    """
    # 1. Obtener Quote (Bid/Ask)
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    if not tick:
        return {"status": "error", "message": f"No se pudo obtener tick para {symbol}"}
    
    # 2. Obtener ltima vela (Situacin actual)
    candles_data = await get_candles(symbol, timeframe, n_candles=1)
    if candles_data.get("status") != "success":
        last_candle = None
    else:
        # Extraer la ltima vela de la lista
        last_candle = candles_data["candles"][-1] if candles_data["candles"] else None

    # 3. Obtener Volatilidad
    vol_data = await get_volatility_metrics(symbol, timeframe)
    
    # 4. Obtener Estado de Cuenta (Para gestin de riesgo)
    account_data = await get_account_metrics()

    snapshot = {
        "status": "success",
        "symbol": symbol,
        "timestamp": time.time(),
        "market": {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread_points": vol_data.get("spread_points", 0),
            "volatility": vol_data.get("volatility_status", "UNKNOWN"),
            "is_market_open": (tick.time > (time.time() - 600)) # Simple check de frescura
        },
        "technical": {
            "timeframe": timeframe,
            "last_candle": last_candle
        },
        "account": {
            "balance": account_data.get("balance"),
            "equity": account_data.get("equity"),
            "margin_level": account_data.get("margin_level"),
            "free_margin_percent": (account_data.get("margin_free", 0) / account_data.get("balance", 1)) * 100 if account_data.get("balance") else 0
        }
    }
    
    return snapshot
