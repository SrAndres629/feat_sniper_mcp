"""
N8N Bridge: LLM Integration for RLAIF Bicameral Architecture
=============================================================
Handles communication between the Neural Network and the n8n LLM Judge.

Features:
- Trade audit requests (SUPERVISED mode)
- Performance reports (periodic)
- Diagnosis requests (RECALIBRATION mode)
- Feedback storage in ChromaDB
"""

import logging
import asyncio
import aiohttp
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger("MT5_Bridge.Services.N8N")


@dataclass
class AuditRequest:
    """Solicitud de auditoría de trade para el LLM."""
    trade_id: str
    symbol: str
    direction: str  # BUY/SELL
    entry_price: float
    proposed_sl: float
    proposed_tp: float
    lot_size: float
    nn_confidence: float
    feat_scores: Dict[str, float]  # FEAT metrics
    energy_map_summary: Dict[str, Any]  # Key points from energy map
    market_context: Dict[str, Any]  # H4 bias, volatility, etc.
    justification: str  # Human-readable reasoning


@dataclass
class AuditResponse:
    """Respuesta del LLM a la solicitud de auditoría."""
    decision: str  # "APPROVE" or "REJECT"
    feedback: str
    suggested_sl: Optional[float] = None
    suggested_tp: Optional[float] = None
    confidence_adjustment: float = 0.0
    reasoning: str = ""


class N8NBridge:
    """
    Puente de comunicación con n8n para el ciclo RLAIF.
    """
    
    N8N_CONFIG_FILE = "data/n8n_config.json"
    FEEDBACK_LOG_FILE = "data/llm_feedback_log.jsonl"
    
    def __init__(self):
        self.webhook_url: Optional[str] = None
        self.api_key: Optional[str] = None
        self.timeout = 30  # seconds
        self.enabled = False
        self._load_config()
    
    def _load_config(self):
        """Carga configuración de n8n desde archivo, variables de entorno, o auto-discovery."""
        # Priority 1: Environment variables
        self.webhook_url = os.getenv("N8N_WEBHOOK_URL")
        self.api_key = os.getenv("N8N_API_KEY")
        
        # Priority 2: Config file
        if not self.webhook_url and os.path.exists(self.N8N_CONFIG_FILE):
            try:
                with open(self.N8N_CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.webhook_url = config.get("webhook_url")
                    self.api_key = config.get("api_key")
                    self.timeout = config.get("timeout", 30)
            except Exception as e:
                logger.warning(f"[N8N] Could not load config: {e}")
        
        # Priority 3: Auto-discovery (intelligent detection)
        if not self.webhook_url:
            self._auto_discover()
        
        self.enabled = bool(self.webhook_url)
        if self.enabled:
            logger.info(f"[N8N] Bridge enabled: {self.webhook_url[:50]}...")
        else:
            logger.warning("[N8N] Bridge disabled - no webhook URL configured")
    
    def _auto_discover(self):
        """
        Auto-discovery de n8n webhook.
        Busca n8n en puertos locales comunes y configura automáticamente.
        """
        import socket
        
        # Common n8n ports/locations
        n8n_candidates = [
            ("localhost", 5678),        # Default n8n
            ("127.0.0.1", 5678),
            ("n8n", 5678),              # Docker service name
            ("n8n-architect-mcp", 8001), # Our custom MCP
        ]
        
        for host, port in n8n_candidates:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    # n8n found! Build webhook URL
                    base_url = f"http://{host}:{port}"
                    
                    # Default webhook path for FEAT SNIPER
                    self.webhook_url = f"{base_url}/webhook/feat-audit"
                    self.n8n_base_url = base_url
                    
                    logger.info(f"[N8N] Auto-discovered at {base_url}")
                    
                    # Save for future use
                    self.save_config(self.webhook_url, self.api_key)
                    return
            except Exception:
                continue
        
        # Last resort: Check if n8n-architect MCP is available
        try:
            # Try to get info from MCP (if available)
            self.webhook_url = "http://localhost:5678/webhook/feat-audit"
            self.n8n_base_url = "http://localhost:5678"
            logger.info("[N8N] Using default n8n location (localhost:5678)")
        except Exception:
            logger.warning("[N8N] Auto-discovery failed - configure manually")
    
    def save_config(self, webhook_url: str, api_key: str = None):
        """Guarda configuración de n8n."""
        os.makedirs(os.path.dirname(self.N8N_CONFIG_FILE) or ".", exist_ok=True)
        config = {
            "webhook_url": webhook_url,
            "api_key": api_key,
            "timeout": self.timeout
        }
        with open(self.N8N_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.enabled = True
        logger.info("[N8N] Configuration saved and bridge enabled")
    
    async def request_audit(self, request: AuditRequest) -> AuditResponse:
        """
        Solicita auditoría del LLM para un trade propuesto.
        
        En modo SUPERVISED, el trade NO se ejecuta sin aprobación.
        """
        if not self.enabled:
            logger.warning("[N8N] Audit skipped - bridge disabled")
            return AuditResponse(
                decision="APPROVE",
                feedback="N8N bridge not configured - auto-approve",
                reasoning="Bridge disabled"
            )
        
        payload = {
            "type": "TRADE_AUDIT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **asdict(request)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        audit_response = AuditResponse(
                            decision=data.get("decision", "REJECT"),
                            feedback=data.get("feedback", "No feedback provided"),
                            suggested_sl=data.get("suggested_sl"),
                            suggested_tp=data.get("suggested_tp"),
                            confidence_adjustment=data.get("confidence_adjustment", 0.0),
                            reasoning=data.get("reasoning", "")
                        )
                        
                        # Log feedback for learning
                        self._log_feedback(request.trade_id, payload, asdict(audit_response))
                        
                        logger.info(f"[N8N] Audit response: {audit_response.decision} - {audit_response.feedback[:50]}...")
                        return audit_response
                    else:
                        logger.error(f"[N8N] Webhook returned {response.status}")
                        return AuditResponse(
                            decision="REJECT",
                            feedback=f"Webhook error: {response.status}",
                            reasoning="HTTP error"
                        )
        
        except asyncio.TimeoutError:
            logger.error("[N8N] Audit request timed out")
            return AuditResponse(
                decision="REJECT",
                feedback="Timeout waiting for LLM response",
                reasoning="Timeout"
            )
        except Exception as e:
            logger.error(f"[N8N] Audit request failed: {e}")
            return AuditResponse(
                decision="REJECT",
                feedback=f"Error: {str(e)}",
                reasoning="Exception"
            )
    
    async def send_performance_report(self, metrics: Dict[str, Any]) -> bool:
        """
        Envía reporte periódico de rendimiento al LLM.
        El LLM puede monitorear sin bloquear operaciones.
        """
        if not self.enabled:
            return False
        
        payload = {
            "type": "PERFORMANCE_REPORT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)  # Shorter timeout for reports
                ) as response:
                    success = response.status == 200
                    if success:
                        logger.debug("[N8N] Performance report sent successfully")
                    return success
        
        except Exception as e:
            logger.warning(f"[N8N] Failed to send performance report: {e}")
            return False
    
    async def request_diagnosis(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solicita diagnóstico del LLM cuando el sistema entra en RECALIBRATION.
        
        El LLM puede:
        - Sugerir cambios de parámetros
        - Solicitar reentrenamiento
        - Invocar herramientas MCP via Antigravity
        """
        if not self.enabled:
            return {"diagnosis": "N8N bridge disabled", "actions": []}
        
        payload = {
            "type": "DIAGNOSIS_REQUEST",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "HIGH",
            **error_context
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)  # Longer timeout for diagnosis
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"[N8N] Diagnosis received: {data.get('summary', 'No summary')[:100]}...")
                        return data
                    else:
                        return {"diagnosis": f"HTTP {response.status}", "actions": []}
        
        except Exception as e:
            logger.error(f"[N8N] Diagnosis request failed: {e}")
            return {"diagnosis": str(e), "actions": []}
    
    def _log_feedback(self, trade_id: str, request: Dict, response: Dict):
        """Guarda feedback del LLM para aprendizaje futuro."""
        os.makedirs(os.path.dirname(self.FEEDBACK_LOG_FILE) or ".", exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": trade_id,
            "request": request,
            "response": response
        }
        
        try:
            with open(self.FEEDBACK_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"[N8N] Could not log feedback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna estado del bridge."""
        return {
            "enabled": self.enabled,
            "webhook_configured": bool(self.webhook_url),
            "timeout": self.timeout
        }


# Singleton global
n8n_bridge = N8NBridge()


# =============================================================================
# MCP-COMPATIBLE ASYNC WRAPPERS
# =============================================================================

async def configure_n8n(webhook_url: str, api_key: str = None) -> Dict[str, Any]:
    """MCP Tool: Configura el bridge de n8n."""
    n8n_bridge.save_config(webhook_url, api_key)
    return {"status": "configured", **n8n_bridge.get_status()}


async def get_n8n_status() -> Dict[str, Any]:
    """MCP Tool: Obtiene estado del bridge n8n."""
    return n8n_bridge.get_status()
