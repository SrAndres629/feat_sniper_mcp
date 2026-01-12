import logging
from typing import Dict, Any, List, Optional
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from app.core.mt5_conn import mt5_conn
from app.models.schemas import IndicatorRequest, ResponseModel, ErrorDetail

logger = logging.getLogger("MT5_Bridge.Skills.Indicators")

# Mapeo de Timeframes (reutilizado)
from app.skills.market import TIMEFRAME_MAP

async def get_technical_indicator(req: IndicatorRequest) -> Dict[str, Any]:
    """
    Calcula indicadores tcnicos usando datos directos de MT5.
    """
    symbol = req.symbol
    tf_str = req.timeframe.upper()
    indicator = req.indicator.upper()
    
    if tf_str not in TIMEFRAME_MAP:
        return {"status": "error", "message": f"Timeframe {tf_str} no vlido."}
    
    mt5_tf = TIMEFRAME_MAP[tf_str]
    
    # Cantidad de velas necesarias (periodo + margen para clculo)
    n_candles = req.period + 100
    
    rates = await mt5_conn.execute(mt5.copy_rates_from_pos, symbol, mt5_tf, 0, n_candles)
    if rates is None or len(rates) < req.period:
        return {"status": "error", "message": "No hay suficientes datos para el indicador."}
    
    df = pd.DataFrame(rates)
    
    result_data = {"symbol": symbol, "indicator": indicator, "timeframe": tf_str}

    try:
        if indicator == "RSI":
            # RSI Simple (Wilder's Smoothing matches MT5 behavior)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0))
            loss = (-delta.where(delta < 0, 0))
            
            # Wilder's Smoothing equivalent to EWM with com=period-1 (alpha=1/period)
            avg_gain = gain.ewm(com=req.period - 1, adjust=False).mean()
            avg_loss = loss.ewm(com=req.period - 1, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            result_data["value"] = float(df['rsi'].iloc[-1])
            
        elif indicator == "MA":
            # Moving Average
            method = req.ma_method or 0 # 0: SMA
            price_type = req.ma_price or 0 # 0: Close
            
            # Usamos pandas para facilidad si es SMA/EMA
            if method == 0:
                df['ma'] = df['close'].rolling(window=req.period).mean()
            else:
                df['ma'] = df['close'].ewm(span=req.period, adjust=False).mean()
                
            result_data["value"] = float(df['ma'].iloc[-1])

        elif indicator == "ATR":
            # ATR (Wilder's Smoothing)
            high_low = df['high'] - df['low']
            high_cp = np.abs(df['high'] - df['close'].shift())
            low_cp = np.abs(df['low'] - df['close'].shift())
            df['tr'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            
            # Use EWM for ATR to match MT5
            df['atr'] = df['tr'].ewm(com=req.period - 1, adjust=False).mean()
            result_data["value"] = float(df['atr'].iloc[-1])

        elif indicator == "MACD":
            fast = 12
            slow = 26
            signal = 9
            exp1 = df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
            df['hist'] = df['macd'] - df['signal']
            result_data["macd"] = float(df['macd'].iloc[-1])
            result_data["signal"] = float(df['signal'].iloc[-1])
            result_data["histogram"] = float(df['hist'].iloc[-1])

        elif indicator == "BOLLINGER":
            df['sma'] = df['close'].rolling(window=req.period).mean()
            df['std'] = df['close'].rolling(window=req.period).std()
            df['upper'] = df['sma'] + (df['std'] * 2)
            df['lower'] = df['sma'] - (df['std'] * 2)
            result_data["mid"] = float(df['sma'].iloc[-1])
            result_data["upper"] = float(df['upper'].iloc[-1])
            result_data["lower"] = float(df['lower'].iloc[-1])

        result_data["status"] = "success"
        return result_data

    except Exception as e:
        logger.error(f"Error calculando {indicator}: {e}")
        return {"status": "error", "message": str(e)}
