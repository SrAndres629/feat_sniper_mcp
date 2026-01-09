import logging
from typing import Dict, Any, Optional
import MetaTrader5 as mt5
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
        "suggestion": "Aumenta la desviación (slippage) permitida o intenta en un momento de menor volatilidad."
    },
    mt5.TRADE_RETCODE_REJECT: {
        "code": "ORDER_REJECTED",
        "message": "La orden fue rechazada por el broker.",
        "suggestion": "Revisa si el volumen es válido o si hay restricciones en el símbolo seleccionado."
    },
    mt5.TRADE_RETCODE_CANCEL: {
        "code": "ORDER_CANCELLED",
        "message": "La orden fue cancelada por el servidor.",
        "suggestion": "Verifica la conexión con el servidor del broker e intenta de nuevo."
    },
    mt5.TRADE_RETCODE_NO_MONEY: {
        "code": "INSUFFICIENT_FUNDS",
        "message": "Margen insuficiente para abrir la posición.",
        "suggestion": "Reduce el volumen del lote o deposita fondos en la cuenta."
    },
    mt5.TRADE_RETCODE_TRADE_DISABLED: {
        "code": "TRADING_DISABLED",
        "message": "El trading está deshabilitado para esta cuenta o símbolo.",
        "suggestion": "Contacta a tu broker o verifica si el mercado está abierto."
    },
    mt5.TRADE_RETCODE_MARKET_CLOSED: {
        "code": "MARKET_CLOSED",
        "message": "El mercado está cerrado actualmente.",
        "suggestion": "Espera a la apertura de la sesión de trading correspondiente."
    },
    mt5.TRADE_RETCODE_INVALID_STOPS: {
        "code": "INVALID_STOPS",
        "message": "Niveles de Stop Loss (SL) o Take Profit (TP) inválidos.",
        "suggestion": "Asegúrate de respetar la distancia mínima de puntos (freeze level) dictada por el broker."
    },
    mt5.TRADE_RETCODE_INVALID_VOLUME: {
        "code": "INVALID_VOLUME",
        "message": "El volumen de la orden no es válido.",
        "suggestion": "Verifica el lote mínimo, el lote máximo y el step de volumen del símbolo."
    }
}

# Mapeo de tipos de órdenes
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
    Instrumentado con OTel y Métricas de Ejecución.
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
                    message="Operación bloqueada por límite de drawdown diario.",
                    suggestion="Espera a que se recupere la equidad o ajusta el parámetro MAX_DAILY_DRAWDOWN_PERCENT."
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

        # 1. VALIDACIÓN INTELIGENTE (SMART GUARDRAILS)
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
                    suggestion="Corrige los parámetros de la orden basados en los límites del broker."
                )
            )

        # 2. VALIDACIÓN DE MARGEN
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

        # 3. CALCULO DE DESVIACIÓN DINÁMICA (SLIPPAGE)
        symbol_info = await mt5_conn.execute(mt5.symbol_info, symbol)
        spread = symbol_info.spread
        dynamic_deviation = max(20, int(spread * 1.5)) 

        # 4. PREPARAR REQUEST
        order_type = ACTION_TO_MT5_TYPE.get(action)
        
        if "LIMIT" in action or "STOP" in action:
            if not price:
                return ResponseModel(
                    status="error",
                    error=ErrorDetail(code="PRICE_REQUIRED", message="El precio es requerido para órdenes pendientes.", suggestion="Envía un 'price'.")
                )
            exec_price = price
        else:
            tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
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

        # 5. EJECUCIÓN
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
                    suggestion="Error crítico de comunicación con MetaTrader 5."
                )
            )

        # 6. MANEJO DE RESULTADO
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            if result.retcode in [mt5.TRADE_RETCODE_ERROR, mt5.TRADE_RETCODE_NO_MONEY]:
                circuit_breaker.record_failure()

            obs_engine.track_order(symbol, action, f"FAILED_{result.retcode}")
            hint = RETCODE_HINTS.get(result.retcode, {
                "code": f"RETCODE_{result.retcode}",
                "message": f"Fallo en la ejecución: {result.comment}",
                "suggestion": "Consulta el log de MT5 para más detalles técnicos."
            })
            
            return ResponseModel(
                status="error",
                error=ErrorDetail(**hint)
            )

        # Éxito
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
