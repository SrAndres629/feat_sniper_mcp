from app.core.mt5_conn import mt5

# MAPEO INTELIGENTE DE RETCODES (SMART ERRORS)
RETCODE_HINTS = {
    mt5.TRADE_RETCODE_REQUOTE: {
        "code": "REQUOTE",
        "message": "El precio ha cambiado.",
        "suggestion": "Aumenta la desviación (slippage) permitida."
    },
    mt5.TRADE_RETCODE_REJECT: {
        "code": "ORDER_REJECTED",
        "message": "La orden fue rechazada por el broker."
    },
    mt5.TRADE_RETCODE_NO_MONEY: {
        "code": "INSUFFICIENT_FUNDS",
        "message": "Margen insuficiente."
    },
    mt5.TRADE_RETCODE_INVALID_STOPS: {
        "code": "INVALID_STOPS",
        "message": "Niveles de Stop Loss (SL) o Take Profit (TP) inválidos."
    },
    mt5.TRADE_RETCODE_INVALID_VOLUME: {
        "code": "INVALID_VOLUME",
        "message": "El volumen de la orden no es válido."
    }
}

TRANSIENT_RETCODES = {mt5.TRADE_RETCODE_REQUOTE}

ACTION_TO_MT5_TYPE = {
    "BUY": mt5.ORDER_TYPE_BUY,
    "SELL": mt5.ORDER_TYPE_SELL,
    "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
    "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
    "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
    "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP
}
