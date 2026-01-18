import asyncio
import logging
import time
import pandas as pd
from typing import Dict, Any, Optional
from app.core.mt5_conn import mt5_conn, mt5
from app.core.config import settings
from app.models.schemas import TradeOrderRequest, ResponseModel, TradeOrderResponse, ErrorDetail
from app.core.validation import OrderValidator
from .models import RETCODE_HINTS, TRANSIENT_RETCODES, ACTION_TO_MT5_TYPE

logger = logging.getLogger("Execution.Engine")

async def send_order(order_data: TradeOrderRequest, urgency_score: float = 0.5) -> ResponseModel[TradeOrderResponse]:
    """
    Ejecuta una orden (Mercado o Pendiente) con validaciones de seguridad avanzadas.
    """
    from app.core.observability import obs_engine, tracer
    start_time = time.time()
    symbol, action, volume = order_data.symbol, order_data.action.upper(), order_data.volume
    price, sl, tp = order_data.price, order_data.sl, order_data.tp

    with tracer.start_as_current_span("mt5_send_order") as span:
        # Pre-checks
        from app.services.circuit_breaker import circuit_breaker
        if not circuit_breaker.can_execute():
            return ResponseModel(status="error", error=ErrorDetail(code="CIRCUIT_OPEN", message="Emergency isolation mode."))

        if order_data.decision_ts:
            age = (time.time() * 1000) - order_data.decision_ts
            if age > settings.DECISION_TTL_MS:
                return ResponseModel(status="error", error=ErrorDetail(code="TTL_EXPIRED", message=f"Decision too old ({age:.0f}ms)"))

        from app.services.risk import risk_engine
        if not await risk_engine.check_drawdown_limit():
            return ResponseModel(status="error", error=ErrorDetail(code="RISK_VETO", message="Daily drawdown limit hit."))

        # Vetoes
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        if symbol_info and hasattr(symbol_info, 'spread') and symbol_info.spread > settings.SPREAD_MAX_PIPS:
            return ResponseModel(status="error", error=ErrorDetail(code="SPREAD_VETO", message="Spread too high."))

        # Sizing
        if volume <= 0 and sl:
            tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
            sl_pts = abs((tick.ask if action == "BUY" else tick.bid) - sl) / symbol_info.point
            volume = await risk_engine.calculate_dynamic_lot(urgency_score, (symbol_info.spread/100.0), symbol, int(sl_pts))

        # Validation
        val, err = await OrderValidator.validate_order(symbol, volume, action, price, sl, tp)
        if not val: return ResponseModel(status="error", error=ErrorDetail(code="VALIDATION_FAILED", message=err))
        
        has_m, m_err = await OrderValidator.validate_margin(symbol, volume, action)
        if not has_m: return ResponseModel(status="error", error=ErrorDetail(code="INSUFFICIENT_MARGIN", message=m_err))

        # Deviation/Routing
        dev = max(20, int(symbol_info.spread * 1.5)) * (3 if urgency_score > settings.URGENCY_THRESHOLD else 1)
        order_type = ACTION_TO_MT5_TYPE.get(action)
        tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
        exec_price = price or (tick.ask if "BUY" in action else tick.bid)
        
        # Filling
        f_mode = mt5.ORDER_FILLING_RETURN
        if "LIMIT" not in action and "STOP" not in action:
            if symbol_info.filling_mode & mt5.SYMBOL_FILLING_IOC: f_mode = mt5.ORDER_FILLING_IOC
            elif symbol_info.filling_mode & mt5.SYMBOL_FILLING_FOK: f_mode = mt5.ORDER_FILLING_FOK

        req = {
            "action": mt5.TRADE_ACTION_DEAL if "LIMIT" not in action and "STOP" not in action else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol, "volume": volume, "type": order_type, "price": exec_price,
            "sl": sl or 0.0, "tp": tp or 0.0, "deviation": dev, "magic": 234000,
            "comment": order_data.comment, "type_time": mt5.ORDER_TIME_GTC, "type_filling": f_mode,
        }

        # Execution with Atomic Retries
        result = None
        is_buy = "BUY" in action
        for attempt in range(1, settings.MAX_ORDER_RETRIES + 1):
            result = await mt5_conn.execute_atomic(_place_order_atomic, symbol, req, is_buy)
            if not result or result.retcode == mt5.TRADE_RETCODE_DONE: break
            if result.retcode in TRANSIENT_RETCODES:
                req["deviation"] = int(dev * (1 + 0.5 * attempt))
                await asyncio.sleep(settings.RETRY_BACKOFF_BASE_MS * (3**(attempt-1)) / 1000)
                continue
            break

        if not result or result.retcode != mt5.TRADE_RETCODE_DONE:
            circuit_breaker.record_failure()
            hint = RETCODE_HINTS.get(result.retcode if result else -1, {"code": "EXEC_FAIL", "message": "Execution failed"})
            return ResponseModel(status="error", error=ErrorDetail(**hint))

        circuit_breaker.record_success()
        obs_engine.track_latency("full_order_cycle", symbol, time.time() - start_time)
        return ResponseModel(status="success", data=TradeOrderResponse(result.order, symbol, action, result.price, result.volume))

def _place_order_atomic(sym, req, buy_flag):
    tick = mt5.symbol_info_tick(sym)
    if not tick: return None
    if "LIMIT" not in req["action"]: req["price"] = tick.ask if buy_flag else tick.bid
    return mt5.order_send(req)
