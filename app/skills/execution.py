import logging
from typing import Dict, Any, Optional
try:
    import MetaTrader5 as mt5
except ImportError:
    from unittest.mock import MagicMock
    mt5 = MagicMock()
from app.core.mt5_conn import mt5_conn
from app.models.schemas import TradeOrderRequest, ResponseModel, TradeOrderResponse, ErrorDetail
from app.core.validation import OrderValidator

logger = logging.getLogger("MT5_Bridge.Skills.Execution")

# =============================================================================
# MAPEO INTELIGENTE DE RETCODES (SMART ERRORS)
# =============================================================================
RETCODE_HINTS = {
    mt5.TRADE_RETCODE_REQUOTE: {
        "code": "REQUOTE",
        "message": "El precio ha cambiado.",
        "suggestion": "Aumenta la desviacin (slippage) permitida o intenta en un momento de menor volatilidad."
    },
    mt5.TRADE_RETCODE_REJECT: {
        "code": "ORDER_REJECTED",
        "message": "La orden fue rechazada por el broker.",
        "suggestion": "Revisa si el volumen es vlido o si hay restricciones en el smbolo seleccionado."
    },
    mt5.TRADE_RETCODE_CANCEL: {
        "code": "ORDER_CANCELLED",
        "message": "La orden fue cancelada por el servidor.",
        "suggestion": "Verifica la conexin con el servidor del broker e intenta de nuevo."
    },
    mt5.TRADE_RETCODE_NO_MONEY: {
        "code": "INSUFFICIENT_FUNDS",
        "message": "Margen insuficiente para abrir la posicin.",
        "suggestion": "Reduce el volumen del lote o deposita fondos en la cuenta."
    },
    mt5.TRADE_RETCODE_TRADE_DISABLED: {
        "code": "TRADING_DISABLED",
        "message": "El trading est deshabilitado para esta cuenta o smbolo.",
        "suggestion": "Contacta a tu broker o verifica si el mercado est abierto."
    },
    mt5.TRADE_RETCODE_MARKET_CLOSED: {
        "code": "MARKET_CLOSED",
        "message": "El mercado est cerrado actualmente.",
        "suggestion": "Espera a la apertura de la sesin de trading correspondiente."
    },
    mt5.TRADE_RETCODE_INVALID_STOPS: {
        "code": "INVALID_STOPS",
        "message": "Niveles de Stop Loss (SL) o Take Profit (TP) invlidos.",
        "suggestion": "Asegrate de respetar la distancia mnima de puntos (freeze level) dictada por el broker."
    },
    mt5.TRADE_RETCODE_INVALID_VOLUME: {
        "code": "INVALID_VOLUME",
        "message": "El volumen de la orden no es vlido.",
        "suggestion": "Verifica el lote mnimo, el lote mximo y el step de volumen del smbolo."
    }
}

# Mapeo de tipos de rdenes
ACTION_TO_MT5_TYPE = {
    "BUY": mt5.ORDER_TYPE_BUY,
    "SELL": mt5.ORDER_TYPE_SELL,
    "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
    "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
    "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
    "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP
}

async def send_order(order_data: TradeOrderRequest) -> ResponseModel[TradeOrderResponse]:
    """
    Ejecuta una orden (Mercado o Pendiente) con validaciones de seguridad avanzadas.
    Instrumentado con OTel y Mtricas de Ejecucin.
    """
    from app.core.observability import obs_engine, tracer
    import time

    start_time = time.time()
    symbol = order_data.symbol
    action = order_data.action.upper()
    volume = order_data.volume
    price = order_data.price
    sl = order_data.sl
    tp = order_data.tp

    with tracer.start_as_current_span("mt5_send_order") as span:
        span.set_attribute("symbol", symbol)
        span.set_attribute("action", action)
        span.set_attribute("volume", volume)

        # 0.0 CIRCUIT BREAKER CHECK
        from app.services.circuit_breaker import circuit_breaker
        if not circuit_breaker.can_execute():
            obs_engine.track_order(symbol, action, "CIRCUIT_OPEN")
            return ResponseModel(
                status="error",
                error=ErrorDetail(code="CIRCUIT_OPEN", message="System is in emergency isolation mode.", suggestion="Check logs for recent failures.")
            )

        # 0.1 LIQUIDITY PRE-FLIGHT (Sniper 2.0)
        from app.skills.liquidity import check_liquidity_preflight
        if not await check_liquidity_preflight(symbol):
            obs_engine.track_order(symbol, action, "LOW_LIQUIDITY")
            return ResponseModel(
                status="error",
                error=ErrorDetail(code="LOW_LIQUIDITY", message=f"Liquidity for {symbol} is below institutional grade.", suggestion="Avoid entries during market roll-over or news.")
            )

        # 0.2 INSTITUTIONAL RISK VETO
        from app.services.risk_engine import risk_engine
        if not await risk_engine.check_drawdown_limit():
            obs_engine.track_order(symbol, action, "RISK_VETO")
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code="RISK_VETO",
                    message="Operacin bloqueada por lmite de drawdown diario.",
                    suggestion="Espera a que se recupere la equidad o ajusta el parmetro MAX_DAILY_DRAWDOWN_PERCENT."
                )
            )

        # Adaptive Volume Calculation (if volume == 0 or not provided)
        if volume <= 0 and sl:
            tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
            exec_price = tick.ask if action == "BUY" else tick.bid
            symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
            sl_points = abs(exec_price - sl) / symbol_info.point
            volume = await risk_engine.get_adaptive_lots(symbol, int(sl_points))
            span.set_attribute("adaptive_volume", volume)

        # 1. VALIDACIN INTELIGENTE (SMART GUARDRAILS)
        is_valid, error_msg = await OrderValidator.validate_order(
            symbol=symbol,
            volume=volume,
            action=action,
            price=price,
            sl=sl,
            tp=tp
        )
        
        if not is_valid:
            obs_engine.track_order(symbol, action, "VALIDATION_FAILED")
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code="VALIDATION_FAILED",
                    message=error_msg,
                    suggestion="Corrige los parmetros de la orden basados en los lmites del broker."
                )
            )

        # 2. VALIDACIN DE MARGEN
        has_margin, margin_err = await OrderValidator.validate_margin(symbol, volume, action)
        if not has_margin:
            obs_engine.track_order(symbol, action, "INSUFFICIENT_MARGIN")
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code="INSUFFICIENT_MARGIN",
                    message=margin_err,
                    suggestion="Reduce el volumen o deposita fondos."
                )
            )

        # 3. CALCULO DE DESVIACIN DINMICA (SLIPPAGE)
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        spread = symbol_info.spread
        dynamic_deviation = max(20, int(spread * 1.5)) 

        # 4. PREPARAR REQUEST
        order_type = ACTION_TO_MT5_TYPE.get(action)
        
        if "LIMIT" in action or "STOP" in action:
            if not price:
                return ResponseModel(
                    status="error",
                    error=ErrorDetail(code="PRICE_REQUIRED", message="El precio es requerido para rdenes pendientes.", suggestion="Enva un 'price'.")
                )
            exec_price = price
        else:
            tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
            # Sniper 2.0: Optimization - If suggest buy at 2000.5, we place limit at 2000.0 (better price)
            # This is done by the caller/agent, but we ensure the price is used if provided.
            if price and price > 0:
                exec_price = price
                order_type = ACTION_TO_MT5_TYPE.get(f"{action}_LIMIT", ACTION_TO_MT5_TYPE.get(action))
            else:
                exec_price = tick.ask if action == "BUY" else tick.bid

        # Detectar Filling Mode
        filling_mode = mt5.ORDER_FILLING_RETURN
        if "LIMIT" not in action and "STOP" not in action:
            s_filling = symbol_info.filling_mode
            if s_filling & mt5.SYMBOL_FILLING_IOC:
                filling_mode = mt5.ORDER_FILLING_IOC
            elif s_filling & mt5.SYMBOL_FILLING_FOK:
                filling_mode = mt5.ORDER_FILLING_FOK
                
        request = {
            "action": mt5.TRADE_ACTION_DEAL if "LIMIT" not in action and "STOP" not in action else mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": exec_price,
            "sl": sl or 0.0,
            "tp": tp or 0.0,
            "deviation": dynamic_deviation,
            "magic": 234000,
            "comment": order_data.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        logger.info(f"Enviando orden {action} de {volume} lots en {symbol} a {exec_price} (Dev: {dynamic_deviation})")

        # 5. EJECUCIN
        result = await mt5_conn.execute(mt5.order_send, request)

        if result is None:
            circuit_breaker.record_failure()
            obs_engine.track_order(symbol, action, "MT5_ERROR")
            error = await mt5_conn.execute(mt5.last_error)
            return ResponseModel(
                status="error",
                error=ErrorDetail(
                    code=f"MT5_INTERNAL_ERROR_{error[0]}",
                    message=error[1],
                    suggestion="Error crtico de comunicacin con MetaTrader 5."
                )
            )

        # 6. MANEJO DE RESULTADO
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode in [mt5.TRADE_RETCODE_ERROR, mt5.TRADE_RETCODE_NO_MONEY]:
                circuit_breaker.record_failure()

            obs_engine.track_order(symbol, action, f"FAILED_{result.retcode}")
            hint = RETCODE_HINTS.get(result.retcode, {
                "code": f"RETCODE_{result.retcode}",
                "message": f"Fallo en la ejecucin: {result.comment}",
                "suggestion": "Consulta el log de MT5 para ms detalles tcnicos."
            })
            
            return ResponseModel(
                status="error",
                error=ErrorDetail(**hint)
            )

        # xito
        circuit_breaker.record_success()
        obs_engine.track_order(symbol, action, "SUCCESS")
        
        duration = time.time() - start_time
        obs_engine.track_latency("full_order_cycle", symbol, duration)
        
        span.set_attribute("ticket", result.order)
        span.set_attribute("exec_price", result.price)

        return ResponseModel(
            status="success",
            data=TradeOrderResponse(
                ticket=result.order,
                symbol=symbol,
                action=action,
                price=result.price,
                volume=result.volume
            )
        )

# =============================================================================
# TWIN-ENGINE HYBRID EXECUTION
# =============================================================================

async def execute_twin_trade(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Protocolo Twin-Entry: Abre 2 rdenes simultneas con diferentes objetivos.
    - Orden A (Scalp): TP corto ($2 target), Magic = MAGIC_SCALP
    - Orden B (Swing): TP largo ($10+ target), Magic = MAGIC_SWING
    """
    from app.core.config import settings
    from app.services.risk_engine import risk_engine
    
    symbol = signal.get("symbol", "XAUUSD")
    direction = signal.get("direction", "BUY").upper()
    confidence = signal.get("confidence", 0)
    
    # Verificar si podemos abrir dual trade
    allocation = await risk_engine.get_capital_allocation()
    
    results = {
        "scalp_ticket": None,
        "swing_ticket": None,
        "mode": "TWIN" if allocation["can_dual"] else "SCALP_ONLY",
        "allocation": allocation
    }
    
    # Obtener precio actual y calcular TPs
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
    
    if not tick or not symbol_info:
        return {"status": "error", "message": "Failed to get market data"}
    
    point = symbol_info.point
    tick_value = symbol_info.trade_tick_value
    
    # Calcular pips para $2 y $10 profit con 0.01 lotes
    # Profit = Pips * Tick_Value * Volume
    # Pips = Target_USD / (Tick_Value * Volume)
    scalp_pips = int(settings.SCALP_TARGET_USD / (tick_value * 0.01)) if tick_value > 0 else 200
    swing_pips = int(settings.SWING_TARGET_USD / (tick_value * 0.01)) if tick_value > 0 else 1000
    
    # SL ajustado para micro cuenta (~15-20 pips)
    sl_pips = 20
    
    exec_price = tick.ask if direction == "BUY" else tick.bid
    
    if direction == "BUY":
        scalp_tp = exec_price + (scalp_pips * point)
        swing_tp = exec_price + (swing_pips * point)
        sl = exec_price - (sl_pips * point)
    else:
        scalp_tp = exec_price - (scalp_pips * point)
        swing_tp = exec_price - (swing_pips * point)
        sl = exec_price + (sl_pips * point)
    
    # --- ORDEN A: SCALP (El Sueldo) ---
    scalp_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": exec_price,
        "sl": sl,
        "tp": scalp_tp,
        "deviation": 20,
        "magic": settings.MAGIC_SCALP,
        "comment": "TWIN_SCALP",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    
    scalp_result = await mt5_conn.execute(mt5.order_send, scalp_request)
    if scalp_result and scalp_result.retcode == mt5.TRADE_RETCODE_DONE:
        results["scalp_ticket"] = scalp_result.order
        logger.info(f" SCALP ENTRY: Ticket {scalp_result.order} | TP: ${settings.SCALP_TARGET_USD}")
    
    # --- ORDEN B: SWING (La Riqueza) --- Solo si hay margen
    if allocation["can_dual"]:
        swing_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": exec_price,
            "sl": sl,
            "tp": swing_tp,
            "deviation": 20,
            "magic": settings.MAGIC_SWING,
            "comment": "TWIN_SWING",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        swing_result = await mt5_conn.execute(mt5.order_send, swing_request)
        if swing_result and swing_result.retcode == mt5.TRADE_RETCODE_DONE:
            results["swing_ticket"] = swing_result.order
            logger.info(f" SWING ENTRY: Ticket {swing_result.order} | TP: ${settings.SWING_TARGET_USD}")
    else:
        logger.warning(" SURVIVAL MODE: Only Scalp opened due to margin constraints")
    
    results["status"] = "success" if results["scalp_ticket"] else "failed"
    return results
