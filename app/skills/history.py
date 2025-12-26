import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import MetaTrader5 as mt5
import pandas as pd
from app.core.mt5_conn import mt5_conn
from app.models.schemas import HistoryRequest, ResponseModel, ErrorDetail

logger = logging.getLogger("MT5_Bridge.Skills.History")

async def get_trade_history(req: HistoryRequest) -> Dict[str, Any]:
    """
    Obtiene el historial de órdenes cerradas y calcula métricas de rendimiento.
    """
    days = req.days
    from_date = datetime.now() - timedelta(days=days)
    to_date = datetime.now()
    
    # Obtener historial de MT5
    history = await mt5_conn.execute(mt5.history_deals_get, from_date, to_date)
    
    if history is None or len(history) == 0:
        return {
            "status": "success",
            "message": "No se encontraron operaciones en el periodo especificado.",
            "metrics": {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_profit": 0
            },
            "deals": []
        }
    
    df = pd.DataFrame(list(history), columns=history[0]._asdict().keys())
    
    # Filtrar solo 'Deals' que son cierres de operaciones (entry=1 es OUT)
    # entry: 0-IN, 1-OUT, 2-INOUT
    df_out = df[df['entry'] != 0].copy()
    
    if df_out.empty:
         return {"status": "success", "metrics": {"total_trades": 0}, "deals": []}

    # Cálculos de Métricas
    total_trades = len(df_out)
    winning_trades = len(df_out[df_out['profit'] > 0])
    losing_trades = len(df_out[df_out['profit'] <= 0])
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    gross_profit = df_out[df_out['profit'] > 0]['profit'].sum()
    gross_loss = abs(df_out[df_out['profit'] < 0]['profit'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
    
    total_profit = df_out['profit'].sum()
    
    # Limpiar deals para respuesta
    deals_list = []
    for _, row in df_out.tail(20).iterrows(): # Solo los últimos 20 para no saturar contexto
        deals_list.append({
            "ticket": int(row['ticket']),
            "symbol": row['symbol'],
            "profit": float(row['profit']),
            "time": datetime.fromtimestamp(row['time']).strftime('%Y-%m-%d %H:%M:%S'),
            "comment": row['comment']
        })
        
    return {
        "status": "success",
        "period_days": days,
        "metrics": {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_profit": round(total_profit, 2)
        },
        "recent_deals": deals_list
    }
