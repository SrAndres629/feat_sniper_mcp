import logging
import sys
from fastmcp import FastMCP
from contextlib import asynccontextmanager

# Configuracin de Logging CRTICA para MCP
# Todo log debe ir a stderr para no romper el protocolo JSON-RPC en stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("MT5_MCP_Server")

import MetaTrader5 as mt5

# Importar lgica de negocio de la app
from app.core.mt5_conn import mt5_conn
from app.skills import (
    market, vision, execution, trade_mgmt, 
    indicators, history, calendar, 
    quant_coder, custom_loader,
    tester, unified_model, remote_compute # Nuevos mdulos
)
from app.models.schemas import (
    MarketDataRequest, TradeOrderRequest, PositionManageRequest, 
    IndicatorRequest, HistoryRequest, CalendarRequest, MQL5CodeRequest
)

# --- RAG & MEMORY IMPORTS ---
import chromadb
from chromadb.utils import embedding_functions
from db_engine import UnifiedModelDB
import os
import datetime

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
        logger.info(f"Seal recibida va ZMQ: {data}")
        # Sincronizar con la nube de forma asncrona (Fire & Forget)
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

# --- RAG INITIALIZATION ---
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
memory_collection = chroma_client.get_or_create_collection(
    name="market_memories", 
    embedding_function=sentence_transformer_ef
)

# Registrar Skills de Indicadores Propios
custom_loader.register_custom_skills(mcp)

# Registrar Skills de Cmputo Remoto (NEXUS)
remote_compute.register_remote_skills(mcp)

# =============================================================================
# PILLAR 1: CRTEX DE COMPILACIN (Code Builder)
# =============================================================================

@mcp.tool()
async def create_and_compile_indicator(name: str, code: str, compile: bool = True):
    """
    Escribe y compila un nuevo indicador .mq5. 
    Retorna el log de compilacin para auto-correccin.
    """
    req = MQL5CodeRequest(name=name, code=code, compile=compile)
    return await quant_coder.create_native_indicator(req)

# =============================================================================
# PILLAR 2: OJO DE HALCN (Data Bridge)
# =============================================================================

@mcp.tool()
async def get_market_snapshot(symbol: str, timeframe: str = "M5"):
    """
    Radiografa completa del mercado: Precio, Vela actual, Volatilidad y Cuenta.
    Ideal para decisiones de alta frecuencia.
    """
    return await market.get_market_snapshot(symbol, timeframe)

@mcp.tool()
async def get_candles(symbol: str, timeframe: str = "M5", n_candles: int = 100):
    """Obtiene velas histricas (OHLCV)."""
    return await market.get_candles(symbol, timeframe, n_candles)

@mcp.tool()
async def get_market_panorama(resize_factor: float = 0.75):
    """Captura visual del grfico."""
    return await vision.capture_panorama(resize_factor=resize_factor)

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
# PILLAR 4: INYECCIN ESTRATGICA (Unified Model)
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
    """Ejecuta rdenes: BUY, SELL, LIMIT, STOP."""
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
    """Eventos econmicos de alto impacto prximos."""
    req = CalendarRequest(importance="HIGH", days_forward=1)
    return await calendar.get_economic_calendar(req)

# =============================================================================
# SYSTEM HEALTH
# =============================================================================

@mcp.resource("signals://live")
async def get_live_signals():
    """Stream de seales ZMQ disponibles."""
    return "Stream Active"

# =============================================================================
# PILLAR 5: INFINITE MEMORY (RAG)
# =============================================================================

@mcp.tool()
async def ingest_memories(days: int = 30):
    """
    Sincroniza el historial de SQLite con la memoria vectorial RAG. 
    Permite al bot recordar eventos pasados.
    """
    logger.info(f"RAG: Ingestando memorias de los ltimos {days} das...")
    db_path = os.path.join(os.path.dirname(__file__), "unified_model.db")
    if not os.path.exists(db_path):
        return "Error: DB local no encontrada para ingesta."

    db = UnifiedModelDB(db_path)
    try:
        narratives = db.get_narrative_history(days=days)
        if not narratives:
            return "No se encontraron eventos significativos para recordar."
        
        batch_ids = [f"mem_{datetime.datetime.now().timestamp()}_{i}" for i in range(len(narratives))]
        memory_collection.add(documents=narratives, ids=batch_ids)
        return f"xito: {len(narratives)} memorias tcticas sincronizadas en el almacn vectorial."
    finally:
        db.close()

@mcp.tool()
async def query_memory(question: str):
    """
    Consulta la memoria histrica usando lenguaje natural.
    Ejemplo: 'Cmo nos fue con la ltima divergencia?'
    """
    logger.info(f"RAG: Consultando memoria sobre: {question}")
    results = memory_collection.query(query_texts=[question], n_results=5)
    
    docs = results.get('documents', [[]])[0]
    if not docs:
        return "No hay recuerdos similares en el almacn vectorial."
    
    return "### Eventos Histricos Relacionados:\n\n" + "\n\n---\n\n".join(docs)


# =============================================================================
# PILLAR 6: RLAIF BICAMERAL ARCHITECTURE (Neural Telemetry)
# =============================================================================

@mcp.tool()
async def get_inference_full_context(trade_id: str = None):
    """
    Expone el 'pensamiento' completo de la Red Neuronal para auditoría del LLM.
    
    Returns:
        - input_tensors: Datos normalizados que entraron al modelo
        - feat_scores: Vectores FEAT (Form, Espacio, Aceleración, Tiempo)
        - prediction: Probabilidad y propuesta de trade
        - justification: Explicación textual del razonamiento
    """
    from app.core.config import settings
    import sys
    import os
    
    # Get current symbol from settings
    symbol = settings.SYMBOL
    
    # Fetch latest market data for context
    try:
        snapshot = await market.get_market_snapshot(symbol, "M5")
    except Exception:
        snapshot = {}
    
    # Get FEAT scores from analytics
    try:
        feat_analysis = await advanced_analytics.calculate_advanced_metrics(symbol)
    except Exception:
        feat_analysis = {}
    
    # Build inference context
    context = {
        "trade_id": trade_id or f"ctx_{datetime.datetime.now().timestamp()}",
        "symbol": symbol,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        
        # INPUT: Normalized tensors
        "input_tensors": {
            "price_data": snapshot.get("candle", {}),
            "volatility": snapshot.get("volatility", {}),
            "account": snapshot.get("account", {})
        },
        
        # PROCESS: FEAT Scores
        "feat_scores": {
            "feat_form": feat_analysis.get("feat_form", 0),
            "feat_space": feat_analysis.get("feat_space", 0),
            "feat_acceleration": feat_analysis.get("feat_acceleration", 0),
            "feat_time": feat_analysis.get("feat_time", 0),
            "feat_index": feat_analysis.get("feat_index", 0),
            "cvd_imbalance": feat_analysis.get("cvd_imbalance", 0),
            "energy_score": feat_analysis.get("energy_score", 0)
        },
        
        # OUTPUT: Prediction (if model is loaded)
        "prediction": {
            "direction": feat_analysis.get("bias", "NEUTRAL"),
            "confidence": feat_analysis.get("confidence", 0.5),
            "proposed_sl_pips": feat_analysis.get("sl_pips", 50),
            "proposed_tp_pips": feat_analysis.get("tp_pips", 100)
        },
        
        # META: Human-readable justification
        "justification": _generate_justification(feat_analysis, snapshot)
    }
    
    return context


def _generate_justification(feat_analysis: dict, snapshot: dict) -> str:
    """Genera explicación textual del razonamiento de la NN."""
    feat_index = feat_analysis.get("feat_index", 0)
    bias = feat_analysis.get("bias", "NEUTRAL")
    confidence = feat_analysis.get("confidence", 0.5)
    
    if feat_index > 0.7:
        strength = "FUERTE"
    elif feat_index > 0.4:
        strength = "MODERADA"
    else:
        strength = "DÉBIL"
    
    justification = f"""
📊 ANÁLISIS PROBABILÍSTICO
==========================
Señal {strength} hacia {bias} (Confianza: {confidence:.1%})

FEAT Index: {feat_index:.2f}
- Forma: {feat_analysis.get('feat_form', 0):.2f} (Estructura de precio)
- Espacio: {feat_analysis.get('feat_space', 0):.2f} (Zonas de liquidez)
- Aceleración: {feat_analysis.get('feat_acceleration', 0):.2f} (Momentum)
- Tiempo: {feat_analysis.get('feat_time', 0):.2f} (Fase de mercado)

CVD Imbalance: {feat_analysis.get('cvd_imbalance', 0):.2f}
Energy Score: {feat_analysis.get('energy_score', 0):.2f}

Volatilidad ATR: {snapshot.get('volatility', {}).get('atr', 'N/A')}
"""
    return justification.strip()


@mcp.tool()
async def get_neural_state():
    """
    Retorna el estado actual del FSM y métricas de rendimiento.
    Usado por n8n para decidir el modo de operación.
    """
    # Import FSM from nexus_control (avoiding circular imports)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    try:
        from nexus_control import performance_tracker, TradingState
        fsm_status = performance_tracker.get_status()
    except ImportError:
        fsm_status = {
            "current_state": "supervised",
            "winrate": 0.5,
            "error": "FSM not initialized"
        }
    
    # Get Vault status
    try:
        from app.services.risk_engine import the_vault
        vault_status = the_vault.get_status()
    except ImportError:
        vault_status = {"error": "Vault not initialized"}
    
    return {
        "fsm": fsm_status,
        "vault": vault_status,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }


@mcp.tool()
async def get_vault_status():
    """
    Retorna estado del sistema de protección de capital (The Vault).
    """
    try:
        from app.services.risk_engine import the_vault
        return the_vault.get_status()
    except ImportError:
        return {"error": "Vault module not available"}


@mcp.tool()
async def store_trade_reflection(
    trade_id: str,
    nn_proposal: dict,
    llm_feedback: dict,
    market_result: dict
):
    """
    Guarda la reflexión de un trade para aprendizaje futuro.
    Indexa en ChromaDB para futuras consultas semánticas.
    """
    # Create narrative for embedding
    narrative = f"""
Trade {trade_id}: 
NN propuso {nn_proposal.get('direction', 'UNKNOWN')} con confianza {nn_proposal.get('confidence', 0):.1%}.
LLM {llm_feedback.get('decision', 'NO_RESPONSE')}: {llm_feedback.get('feedback', 'Sin feedback')}.
Resultado: {'GANANCIA' if market_result.get('profit', 0) > 0 else 'PÉRDIDA'} de ${market_result.get('profit', 0):.2f}.
"""
    
    # Store in ChromaDB
    doc_id = f"refl_{trade_id}_{datetime.datetime.now().timestamp()}"
    memory_collection.add(
        documents=[narrative.strip()],
        ids=[doc_id],
        metadatas=[{
            "trade_id": trade_id,
            "nn_direction": nn_proposal.get('direction'),
            "llm_decision": llm_feedback.get('decision'),
            "profit": market_result.get('profit', 0)
        }]
    )
    
    logger.info(f"[RLAIF] Trade reflection stored: {trade_id}")
    return {"status": "stored", "doc_id": doc_id}


# =============================================================================
# PILLAR 7: ANTIGRAVITY BRIDGE (Code Improvement Requests)
# =============================================================================

@mcp.tool()
async def request_code_change(
    file_path: str,
    description: str,
    change_type: str = "improvement",
    suggested_code: str = None,
    urgency: str = "normal"
):
    """
    Permite a n8n/LLM solicitar cambios de código.
    
    El agente Antigravity monitorea este archivo para ejecutar cambios.
    
    Args:
        file_path: Ruta relativa al archivo a modificar
        description: Descripción del cambio requerido
        change_type: Tipo de cambio (bug_fix, improvement, refactor, feature)
        suggested_code: Código sugerido (opcional)
        urgency: Nivel de urgencia (low, normal, high, critical)
    
    Returns:
        Ticket de solicitud para tracking
    """
    import uuid
    import json
    import os
    
    request_id = f"CR-{uuid.uuid4().hex[:8].upper()}"
    
    request = {
        "id": request_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "file_path": file_path,
        "description": description,
        "change_type": change_type,
        "suggested_code": suggested_code,
        "urgency": urgency,
        "status": "pending",
        "source": "n8n_llm_supervisor"
    }
    
    # Save to pending requests file (Antigravity monitors this)
    requests_file = os.path.join("data", "code_change_requests.jsonl")
    os.makedirs(os.path.dirname(requests_file), exist_ok=True)
    
    with open(requests_file, "a") as f:
        f.write(json.dumps(request) + "\n")
    
    logger.info(f"[ANTIGRAVITY] Code change request created: {request_id}")
    
    return {
        "status": "queued",
        "request_id": request_id,
        "message": f"Change request {request_id} queued for Antigravity agent",
        "file": file_path,
        "urgency": urgency
    }


@mcp.tool()
async def get_pending_code_changes():
    """
    Lista solicitudes de cambio de código pendientes.
    Útil para que Antigravity poll las solicitudes.
    """
    import json
    import os
    
    requests_file = os.path.join("data", "code_change_requests.jsonl")
    
    if not os.path.exists(requests_file):
        return {"pending": [], "count": 0}
    
    pending = []
    with open(requests_file, "r") as f:
        for line in f:
            try:
                req = json.loads(line)
                if req.get("status") == "pending":
                    pending.append(req)
            except:
                continue
    
    return {
        "pending": pending,
        "count": len(pending)
    }


@mcp.tool()
async def complete_code_change(request_id: str, success: bool, result: str = None):
    """
    Marca una solicitud de cambio como completada.
    Llamado por Antigravity después de aplicar el cambio.
    """
    import json
    import os
    
    requests_file = os.path.join("data", "code_change_requests.jsonl")
    completed_file = os.path.join("data", "code_change_completed.jsonl")
    
    if not os.path.exists(requests_file):
        return {"error": "No requests file found"}
    
    # Read all requests
    all_requests = []
    target_request = None
    
    with open(requests_file, "r") as f:
        for line in f:
            try:
                req = json.loads(line)
                if req.get("id") == request_id:
                    req["status"] = "completed" if success else "failed"
                    req["result"] = result
                    req["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    target_request = req
                    
                    # Log to completed file
                    with open(completed_file, "a") as cf:
                        cf.write(json.dumps(req) + "\n")
                else:
                    all_requests.append(req)
            except:
                continue
    
    # Rewrite pending file without the completed request
    with open(requests_file, "w") as f:
        for req in all_requests:
            if req.get("status") == "pending":
                f.write(json.dumps(req) + "\n")
    
    if target_request:
        logger.info(f"[ANTIGRAVITY] Code change {request_id}: {'SUCCESS' if success else 'FAILED'}")
        return {"status": "updated", "request": target_request}
    
    return {"error": f"Request {request_id} not found"}


if __name__ == "__main__":
    mcp.run()
