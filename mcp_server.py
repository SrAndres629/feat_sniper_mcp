import logging
import sys
from fastmcp import FastMCP
from contextlib import asynccontextmanager

# Configuración de Logging CRÍTICA para MCP
# Todo log debe ir a stderr para no romper el protocolo JSON-RPC en stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("MT5_MCP_Server")

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logger.warning("MetaTrader5 module not found found. Running in Linux/Docker mode? Some local-only features will be disabled.")


# Importar lógica de negocio de la app
from app.core.mt5_conn import mt5_conn
from app.skills import (
    market, vision, execution, trade_mgmt, 
    indicators, history, calendar, 
    quant_coder, custom_loader,
    tester, unified_model, remote_compute # Nuevos módulos
)
from app.models.schemas import (
    MarketDataRequest, TradeOrderRequest, PositionManageRequest, 
    IndicatorRequest, HistoryRequest, CalendarRequest, MQL5CodeRequest
)

from app.core.zmq_bridge import zmq_bridge
from app.skills.advanced_analytics import advanced_analytics
from app.services.supabase_sync import supabase_sync

@asynccontextmanager
async def app_lifespan(server: FastMCP):
    """Maneja el ciclo de vida institucional: MT5 + ZMQ Bridge."""
    logger.info("Iniciando infraestructura institucional...")
    await mt5_conn.startup()
    
    # Iniciar ZMQ Bridge para Streaming de baja latencia
    async def on_signal(data):
        logger.info(f"Señal recibida vía ZMQ: {data}")
        # Sincronizar con la nube de forma asíncrona (Fire & Forget)
        import asyncio
        asyncio.create_task(supabase_sync.log_signal(data))
        
    await zmq_bridge.start(on_signal)
    
    try:
        yield
    finally:
        logger.info("Cerrando infraestructura...")
        await zmq_bridge.stop()
        await mt5_conn.shutdown()

# Inicializar FastMCP
mcp = FastMCP("MT5_Neural_Sentinel", lifespan=app_lifespan)

# Registrar Skills de Indicadores Propios
custom_loader.register_custom_skills(mcp)

# Registrar Skills de Cómputo Remoto (NEXUS)
remote_compute.register_remote_skills(mcp)

# =============================================================================
# PILLAR 1: CÓRTEX DE COMPILACIÓN (Code Builder)
# =============================================================================

@mcp.tool()
async def create_and_compile_indicator(name: str, code: str, compile: bool = True):
    """
    Escribe y compila un nuevo indicador .mq5. 
    Retorna el log de compilación para auto-corrección.
    """
    req = MQL5CodeRequest(name=name, code=code, compile=compile)
    return await quant_coder.create_native_indicator(req)

# =============================================================================
# PILLAR 2: OJO DE HALCÓN (Data Bridge)
# =============================================================================

@mcp.tool()
async def get_market_snapshot(symbol: str, timeframe: str = "M5"):
    """
    Radiografía completa del mercado: Precio, Vela actual, Volatilidad y Cuenta.
    Ideal para decisiones de alta frecuencia.
    """
    return await market.get_market_snapshot(symbol, timeframe)

@mcp.tool()
async def get_candles(symbol: str, timeframe: str = "M5", n_candles: int = 100):
    """Obtiene velas históricas (OHLCV)."""
    return await market.get_candles(symbol, timeframe, n_candles)

@mcp.tool()
async def get_market_panorama(resize_factor: float = 0.75):
    """Captura visual del gráfico."""
    return await vision.capture_panorama(resize_factor=resize_factor)

@mcp.tool()
async def get_trade_decision(symbol: str, timeframe: str = "M5"):
    """
    Decisión de trading integrada para N8N.
    
    Combina ML (GBM/LSTM) + FEAT Score + FSM State + Indicadores
    para generar una señal unificada con contexto completo.
    
    Returns:
        JSON con signal, confidence, market_state, data_context
    """
    from datetime import datetime
    
    # 1. Obtener snapshot del mercado
    snapshot = await market.get_market_snapshot(symbol, timeframe)
    
    # 2. Obtener predicción ML si disponible
    ml_prediction = None
    try:
        from app.ml.ml_engine import ml_engine
        features = {
            "close": snapshot.get("price", {}).get("bid", 0),
            "open": snapshot.get("current_candle", {}).get("open", 0),
            "high": snapshot.get("current_candle", {}).get("high", 0),
            "low": snapshot.get("current_candle", {}).get("low", 0),
            "volume": snapshot.get("current_candle", {}).get("volume", 0),
            "rsi": 50.0,  # Default, se actualiza con indicadores reales
            "ema_fast": snapshot.get("price", {}).get("bid", 0),
            "ema_slow": snapshot.get("price", {}).get("bid", 0),
            "ema_spread": 0,
            "feat_score": 0.0,
            "fsm_state": 0,
            "atr": snapshot.get("volatility", {}).get("atr", 0.001),
            "compression": 0.5,
            "liquidity_above": 0,
            "liquidity_below": 0
        }
        ml_prediction = ml_engine.ensemble_predict(features)
    except Exception as e:
        logger.warning(f"ML prediction unavailable: {e}")
    
    # 3. Construir señal
    confidence = 0.5
    signal = "WAIT"
    
    if ml_prediction and ml_prediction.get("p_win") is not None:
        p_win = ml_prediction["p_win"]
        confidence = abs(p_win - 0.5) * 2
        
        if p_win > 0.65:
            signal = "BUY"
        elif p_win < 0.35:
            signal = "SELL"
        else:
            signal = "WAIT"
            
        # Penalizar si hay anomalía
        if ml_prediction.get("is_anomaly"):
            confidence *= 0.5
            signal = "WAIT"  # No operar en manipulación
    
    # 4. Determinar estado de mercado (FSM)
    volatility = snapshot.get("volatility", {})
    vol_status = volatility.get("volatility_status", "NORMAL")
    
    if vol_status == "EXTREME":
        market_state = "MANIPULATION"
    elif vol_status == "HIGH":
        market_state = "EXPANSION"
    elif vol_status == "LOW":
        market_state = "ACCUMULATION"
    else:
        market_state = "DISTRIBUTION"
    
    # 5. Construir contexto para N8N
    data_context = {
        "price": snapshot.get("price", {}),
        "volatility": {
            "atr": volatility.get("atr"),
            "atr_pips": volatility.get("atr_pips"),
            "status": vol_status,
            "spread_points": volatility.get("spread_points")
        },
        "current_candle": snapshot.get("current_candle", {}),
        "ml_source": ml_prediction.get("source") if ml_prediction else "NONE",
        "is_anomaly": ml_prediction.get("is_anomaly", False) if ml_prediction else False
    }
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": round(confidence, 3),
        "market_state": market_state,
        "data_context": data_context,
        "timestamp": datetime.utcnow().isoformat(),
        "execution_enabled": ml_prediction.get("execution_enabled", False) if ml_prediction else False
    }

# =============================================================================
# PILLAR 3: SIMULADOR DE SOMBRAS (Backtest Automator)
# =============================================================================

@mcp.tool()
async def run_shadow_backtest(
    expert_name: str, 
    symbol: str, 
    period: str = "H1", 
    days: int = 30,
    deposit: int = 10000
):
    """
    Ejecuta una prueba de estrategia en segundo plano usando el Strategy Tester de MT5.
    """
    # Calcular fechas
    import datetime
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    return await tester.run_strategy_test(
        expert_name=expert_name,
        symbol=symbol,
        period=period,
        date_from=start_date.strftime("%Y.%m.%d"),
        date_to=end_date.strftime("%Y.%m.%d"),
        deposit=deposit
    )

# =============================================================================
# PILLAR 4: INYECCIÓN ESTRATÉGICA (Unified Model)
# =============================================================================

@mcp.tool()
async def query_unified_brain(sql_query: str):
    """
    Consulta la base de datos de conocimiento unificado (Unified Model).
    Solo permite SELECT.
    """
    return await unified_model.unified_db.query_custom_sql(sql_query)

# =============================================================================
# EXECUTION & MANAGEMENT SKILLS
# =============================================================================

@mcp.tool()
async def execute_trade(symbol: str, action: str, volume: float, price: float = None, sl: float = None, tp: float = None, comment: str = "MCP_Order"):
    """Ejecuta órdenes: BUY, SELL, LIMIT, STOP."""
    req = TradeOrderRequest(symbol=symbol, action=action, volume=volume, price=price, sl=sl, tp=tp, comment=comment)
    result = await execution.send_order(req)
    return result.dict()

@mcp.tool()
async def manage_trade(ticket: int, action: str, volume: float = None, sl: float = None, tp: float = None):
    """Gestiona posiciones: CLOSE, MODIFY_SL_TP, DELETE."""
    req = PositionManageRequest(ticket=ticket, action=action, volume=volume, sl=sl, tp=tp)
    result = await trade_mgmt.manage_position(req)
    return result.dict()

@mcp.tool()
async def get_account_status():
    """Estado financiero de la cuenta."""
    return await market.get_account_metrics()

@mcp.tool()
async def get_economic_calendar():
    """Eventos económicos de alto impacto próximos."""
    req = CalendarRequest(importance="HIGH", days_forward=1)
    return await calendar.get_economic_calendar(req)

# =============================================================================
# PILLAR 5: MEMORIA INFINITA (RAG Memory)
# =============================================================================

@mcp.tool()
async def remember(text: str, category: str = "general"):
    """
    Almacena información en memoria permanente RAG.
    Las memorias persisten entre reinicios del contenedor.
    
    Categorías sugeridas: analysis, trade, news, pattern, lesson, config
    """
    from app.services.rag_memory import rag_memory
    doc_id = rag_memory.store(text, category=category)
    return {"status": "stored", "id": doc_id, "category": category}

@mcp.tool()
async def recall(query: str, limit: int = 5, category: str = None):
    """
    Busca información relevante en memoria RAG usando búsqueda semántica.
    Retorna los documentos más similares a la query.
    """
    from app.services.rag_memory import rag_memory
    results = rag_memory.search(query, k=limit, category=category)
    return {
        "query": query,
        "count": len(results),
        "memories": results
    }

@mcp.tool()
async def forget(category: str):
    """
    Elimina todas las memorias de una categoría específica.
    Útil para limpiar información obsoleta.
    """
    from app.services.rag_memory import rag_memory
    count = rag_memory.forget(category=category)
    return {"status": "deleted", "count": count, "category": category}

@mcp.tool()
async def memory_stats():
    """
    Obtiene estadísticas de la memoria RAG.
    """
    from app.services.rag_memory import rag_memory
    return {
        "total_memories": rag_memory.count(),
        "categories": rag_memory.get_categories()
    }

# =============================================================================
# PILLAR 6: SYSTEM ADMIN (Auto-Diagnóstico)
# =============================================================================

@mcp.tool()
async def system_check():
    """
    Revisa la salud del servidor (CPU, RAM, Disco).
    Útil para auto-diagnóstico antes de operaciones pesadas.
    """
    from app.skills.system_ops import system_health_check
    return await system_health_check()

@mcp.tool()
async def system_environment():
    """
    Obtiene información del entorno de ejecución.
    """
    from app.skills.system_ops import get_environment_info
    return await get_environment_info()

@mcp.tool()
async def system_cleanup():
    """
    Limpia caches de Python para liberar memoria.
    """
    from app.skills.system_ops import cleanup_cache
    return await cleanup_cache()

# =============================================================================
# PILLAR 7: QUANTUM LEAP ML (Stochastic Predictions)
# =============================================================================

@mcp.tool()
async def ml_predict(features: dict):
    """
    Genera predicción ML usando ensemble GBM+LSTM.
    Shadow Mode por defecto (no ejecuta, solo predice).
    
    Args:
        features: Dict con close, rsi, ema_fast, ema_slow, volume, feat_score, fsm_state, etc.
    """
    from app.ml.ml_engine import predict
    return await predict(features)

@mcp.tool()
async def ml_status():
    """
    Estado del sistema ML: modelos cargados, modo, etc.
    """
    from app.ml.ml_engine import get_ml_status
    return await get_ml_status()

@mcp.tool()
async def ml_collect_sample(symbol: str, candle: dict, indicators: dict):
    """
    Recolecta muestra para training dataset con Oracle labeling.
    
    Args:
        symbol: Par de divisas
        candle: Dict con open, high, low, close, volume
        indicators: Dict con RSI, EMAs, FEAT score, FSM state, etc.
    """
    from app.ml.data_collector import collect_sample
    return await collect_sample(symbol, candle, indicators)

@mcp.tool()
async def ml_train():
    """
    Entrena modelos GBM y LSTM si hay datos suficientes.
    Requiere 1000+ muestras etiquetadas.
    """
    from app.ml.train_models import train_all
    return train_all()

@mcp.tool()
async def ml_enable_execution(enable: bool = True):
    """
    (PELIGROSO) Activa/desactiva ejecución real de órdenes.
    Solo usar después de verificar predicciones en Shadow Mode.
    """
    from app.ml.ml_engine import enable_execution
    return await enable_execution(enable)

# =============================================================================
# SYSTEM RESOURCES
# =============================================================================

@mcp.resource("signals://live")
async def get_live_signals():
    """Stream de señales ZMQ disponibles."""
    return "Stream Active"

if __name__ == "__main__":
    import os
    import sys
    
    # Detectar si estamos en Docker/Linux (sin TTY) o en Windows (con TTY)
    # En Docker usamos HTTP para que el servidor persista
    # En Windows local usamos STDIO para integración directa con IDE
    is_docker = not sys.stdin.isatty() or os.environ.get("DOCKER_MODE", "").lower() == "true"
    
    if is_docker:
        logger.info("🐳 Modo Docker detectado - Iniciando servidor SSE en puerto 8000...")
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    else:
        logger.info("🖥️ Modo Windows detectado - Iniciando servidor STDIO...")
        mcp.run()
