"""
HEALTH SENTINEL - Post-Boot Reliability Validation
===================================================
Sanity Check profundo después del arranque del sistema.

[SENIOR SRE] Este módulo es la "Última Línea de Defensa" antes de que el sistema
empiece a procesar datos reales del mercado. Valida que todos los subsistemas
estén operativos y coherentes.

Features:
- Validación de endpoints con esquema JSON
- Verificación de readiness de sub-módulos
- Benchmark de latencia inicial (TTFB)
- Exponential backoff para reintentos
- Reporte visual de salud
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger("feat.core.health_sentinel")


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_HEALTH_ENDPOINT = "http://127.0.0.1:8000/health"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRIES = 5
LATENCY_WARNING_THRESHOLD_MS = 50


# =============================================================================
# HEALTH STATES
# =============================================================================

class HealthState(Enum):
    """Estados de salud posibles."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SubsystemState(Enum):
    """Estados de subsistemas."""
    READY = "ready"
    WARMING_UP = "warming_up"
    NOT_READY = "not_ready"
    ERROR = "error"


@dataclass
class SubsystemCheck:
    """Resultado de verificación de un subsistema."""
    name: str
    state: SubsystemState
    message: str
    latency_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class HealthReport:
    """Reporte completo de salud del sistema."""
    overall_state: HealthState
    timestamp: str
    subsystems: List[SubsystemCheck]
    endpoint_latency_ms: float
    warnings: List[str]
    is_ready_for_trading: bool


# =============================================================================
# HEALTH SENTINEL - Singleton
# =============================================================================

class HealthSentinel:
    """
    Sentinel de Salud Post-Boot.
    
    ¿Por qué validación post-boot?
    ------------------------------
    Un servidor que responde 200 OK pero tiene el motor de inferencia
    congelado es un "Zombie". Los zombies liquidan cuentas.
    
    El Sentinel verifica:
    1. Que el servidor esté vivo (connectivity)
    2. Que los subsistemas estén ready (ML, Physics, RAG)
    3. Que la latencia sea aceptable (TTFB)
    """
    
    _instance: Optional['HealthSentinel'] = None
    
    def __new__(cls) -> 'HealthSentinel':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._last_report: Optional[HealthReport] = None
        self._httpx_available = False
        
        # Intentar import de httpx para requests async
        try:
            import httpx
            self._httpx_available = True
        except ImportError:
            logger.warning("[SENTINEL] httpx not available - using synchronous requests")
        
        logger.info("[SENTINEL] HealthSentinel initialized")
    
    async def _request_with_retry(self, url: str, timeout: float = 5.0) -> Tuple[Optional[Dict], float, Optional[str]]:
        """
        Realiza request con exponential backoff.
        
        Returns:
            (response_json, latency_ms, error_message)
        """
        if self._httpx_available:
            import httpx
            
            for attempt in range(MAX_RETRIES):
                try:
                    start = time.perf_counter()
                    
                    async with httpx.AsyncClient() as client:
                        response = await client.get(url, timeout=timeout)
                    
                    latency_ms = (time.perf_counter() - start) * 1000
                    
                    if response.status_code == 200:
                        return response.json(), latency_ms, None
                    else:
                        error = f"HTTP {response.status_code}"
                        
                except httpx.TimeoutException:
                    error = "Request timeout"
                except httpx.ConnectError:
                    error = "Connection refused"
                except Exception as e:
                    error = str(e)
                
                # Exponential backoff: 1s, 2s, 4s, 8s...
                wait_time = 2 ** attempt
                logger.warning(f"[SENTINEL] Attempt {attempt + 1} failed: {error}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            
            return None, 0, f"Max retries exceeded: {error}"
        
        else:
            # Fallback a requests síncronos
            import urllib.request
            import json
            
            for attempt in range(MAX_RETRIES):
                try:
                    start = time.perf_counter()
                    
                    with urllib.request.urlopen(url, timeout=timeout) as response:
                        data = json.loads(response.read().decode())
                    
                    latency_ms = (time.perf_counter() - start) * 1000
                    return data, latency_ms, None
                    
                except Exception as e:
                    error = str(e)
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
            
            return None, 0, f"Max retries exceeded: {error}"
    
    def _validate_health_schema(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Valida que la respuesta del endpoint tenga el esquema esperado.
        
        Esquema esperado:
        {
            "status": "online",
            "version": "x.x",
            "bridge_connected": true
        }
        
        Returns:
            (is_valid, missing_fields)
        """
        required_fields = ["status"]
        missing = [f for f in required_fields if f not in data]
        
        if missing:
            return False, missing
        
        # Validar valores
        warnings = []
        if data.get("status") != "online" and data.get("status") != "ok":
            warnings.append(f"Unexpected status: {data.get('status')}")
        
        if data.get("bridge_connected") is False:
            warnings.append("MT5 bridge not connected")
        
        return len(missing) == 0, warnings
    
    async def check_endpoint_health(self, url: str = DEFAULT_HEALTH_ENDPOINT) -> SubsystemCheck:
        """Verifica conectividad del endpoint principal."""
        data, latency_ms, error = await self._request_with_retry(url)
        
        if error:
            return SubsystemCheck(
                name="MCP Server",
                state=SubsystemState.ERROR,
                message=error,
                latency_ms=latency_ms
            )
        
        is_valid, warnings = self._validate_health_schema(data)
        
        if is_valid and not warnings:
            return SubsystemCheck(
                name="MCP Server",
                state=SubsystemState.READY,
                message="Endpoint healthy",
                latency_ms=latency_ms,
                metadata=data
            )
        else:
            return SubsystemCheck(
                name="MCP Server",
                state=SubsystemState.DEGRADED if is_valid else SubsystemState.NOT_READY,
                message="; ".join(warnings) if warnings else "Schema validation failed",
                latency_ms=latency_ms,
                metadata=data
            )
    
    async def check_ml_readiness(self) -> SubsystemCheck:
        """Verifica que el motor ML esté cargado y listo."""
        try:
            from app.ml.ml_engine import ml_engine
            
            status = ml_engine.get_status()
            
            # Verificar componentes críticos
            if not status.get("anomaly_fitted", False):
                return SubsystemCheck(
                    name="ML Engine",
                    state=SubsystemState.WARMING_UP,
                    message="Anomaly detector not fitted yet",
                    metadata=status
                )
            
            return SubsystemCheck(
                name="ML Engine",
                state=SubsystemState.READY,
                message=f"V{status.get('v', 'unknown')} - {len(status.get('symbols_registered', []))} symbols",
                metadata=status
            )
            
        except ImportError:
            return SubsystemCheck(
                name="ML Engine",
                state=SubsystemState.NOT_READY,
                message="MLEngine module not available"
            )
        except Exception as e:
            return SubsystemCheck(
                name="ML Engine",
                state=SubsystemState.ERROR,
                message=str(e)
            )
    
    async def check_physics_engine(self) -> SubsystemCheck:
        """Verifica que el motor de física esté operativo."""
        try:
            from app.skills.market_physics import market_physics, MarketPhysics
            
            # Verificar constantes P0-1
            if hasattr(MarketPhysics, 'MIN_DELTA_T'):
                return SubsystemCheck(
                    name="Physics Engine",
                    state=SubsystemState.READY,
                    message=f"MIN_DELTA_T={MarketPhysics.MIN_DELTA_T}s",
                    metadata={"window_size": market_physics.window_size}
                )
            else:
                return SubsystemCheck(
                    name="Physics Engine",
                    state=SubsystemState.DEGRADED,
                    message="P0-1 fix not applied"
                )
            
        except ImportError:
            return SubsystemCheck(
                name="Physics Engine",
                state=SubsystemState.NOT_READY,
                message="MarketPhysics module not available"
            )
        except Exception as e:
            return SubsystemCheck(
                name="Physics Engine",
                state=SubsystemState.ERROR,
                message=str(e)
            )
    
    async def check_feat_gates(self) -> SubsystemCheck:
        """Verifica que los FEAT Gates estén disponibles."""
        try:
            from app.services.spread_filter import spread_filter
            from app.services.volatility_guard import volatility_guard
            
            if spread_filter and volatility_guard:
                return SubsystemCheck(
                    name="FEAT Gates",
                    state=SubsystemState.READY,
                    message="SpreadFilter + VolatilityGuard active"
                )
            else:
                return SubsystemCheck(
                    name="FEAT Gates",
                    state=SubsystemState.DEGRADED,
                    message="Some gates not available"
                )
            
        except ImportError as e:
            return SubsystemCheck(
                name="FEAT Gates",
                state=SubsystemState.NOT_READY,
                message=f"Import error: {e}"
            )
    
    async def run_full_diagnostic(self, health_url: str = DEFAULT_HEALTH_ENDPOINT) -> HealthReport:
        """
        Ejecuta diagnóstico completo del sistema.
        
        Returns:
            HealthReport con estado de todos los subsistemas
        """
        logger.info("[SENTINEL] Starting full diagnostic...")
        
        subsystems = []
        warnings = []
        
        # 1. Endpoint health (con reintentos)
        endpoint_check = await self.check_endpoint_health(health_url)
        subsystems.append(endpoint_check)
        
        # 2. ML Engine
        ml_check = await self.check_ml_readiness()
        subsystems.append(ml_check)
        
        # 3. Physics Engine
        physics_check = await self.check_physics_engine()
        subsystems.append(physics_check)
        
        # 4. FEAT Gates
        gates_check = await self.check_feat_gates()
        subsystems.append(gates_check)
        
        # Calcular estado global
        error_count = sum(1 for s in subsystems if s.state == SubsystemState.ERROR)
        not_ready_count = sum(1 for s in subsystems if s.state == SubsystemState.NOT_READY)
        degraded_count = sum(1 for s in subsystems if s.state == SubsystemState.DEGRADED)
        
        if error_count > 0:
            overall_state = HealthState.UNHEALTHY
        elif not_ready_count > 0 or degraded_count > 1:
            overall_state = HealthState.DEGRADED
        else:
            overall_state = HealthState.HEALTHY
        
        # Verificar latencia
        if endpoint_check.latency_ms and endpoint_check.latency_ms > LATENCY_WARNING_THRESHOLD_MS:
            warnings.append(f"High initial latency: {endpoint_check.latency_ms:.1f}ms > {LATENCY_WARNING_THRESHOLD_MS}ms")
        
        # Crear reporte
        report = HealthReport(
            overall_state=overall_state,
            timestamp=datetime.now().isoformat(),
            subsystems=subsystems,
            endpoint_latency_ms=endpoint_check.latency_ms or 0,
            warnings=warnings,
            is_ready_for_trading=overall_state == HealthState.HEALTHY
        )
        
        self._last_report = report
        
        return report
    
    def print_health_card(self, report: Optional[HealthReport] = None) -> None:
        """Imprime reporte visual de salud."""
        r = report or self._last_report
        
        if r is None:
            print("[SENTINEL] No health report available")
            return
        
        # Símbolos
        state_symbols = {
            SubsystemState.READY: "✅",
            SubsystemState.WARMING_UP: "⏳",
            SubsystemState.DEGRADED: "⚠️",
            SubsystemState.NOT_READY: "❌",
            SubsystemState.ERROR: "💀"
        }
        
        overall_symbols = {
            HealthState.HEALTHY: "🟢 HEALTHY",
            HealthState.DEGRADED: "🟡 DEGRADED",
            HealthState.UNHEALTHY: "🔴 UNHEALTHY",
            HealthState.UNKNOWN: "⚪ UNKNOWN"
        }
        
        print("\n" + "=" * 60)
        print("  FEAT SNIPER - Health Report")
        print("=" * 60)
        print(f"  Status     : {overall_symbols.get(r.overall_state, 'UNKNOWN')}")
        print(f"  Timestamp  : {r.timestamp}")
        print(f"  Latency    : {r.endpoint_latency_ms:.1f}ms")
        print(f"  Trading OK : {'YES ✅' if r.is_ready_for_trading else 'NO ❌'}")
        print("-" * 60)
        print("  Subsystems:")
        
        for sub in r.subsystems:
            symbol = state_symbols.get(sub.state, "?")
            latency_str = f" ({sub.latency_ms:.1f}ms)" if sub.latency_ms else ""
            print(f"    {symbol} {sub.name}: {sub.message}{latency_str}")
        
        if r.warnings:
            print("-" * 60)
            print("  Warnings:")
            for w in r.warnings:
                print(f"    ⚠️ {w}")
        
        print("=" * 60)
        
        if r.is_ready_for_trading:
            print("  🚀 ALL SYSTEMS GO - Ready for Shadow Deployment")
        else:
            print("  🛑 BOOT FAILED - Do not proceed with trading")
        
        print("=" * 60 + "\n")
    
    def get_status(self) -> Dict:
        """Retorna estado para diagnóstico."""
        if self._last_report is None:
            return {"has_report": False}
        
        r = self._last_report
        return {
            "has_report": True,
            "overall_state": r.overall_state.value,
            "is_ready_for_trading": r.is_ready_for_trading,
            "subsystems_count": len(r.subsystems),
            "warnings_count": len(r.warnings),
            "endpoint_latency_ms": r.endpoint_latency_ms
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

health_sentinel = HealthSentinel()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_boot_diagnostic(health_url: str = DEFAULT_HEALTH_ENDPOINT, print_report: bool = True) -> bool:
    """
    Ejecuta diagnóstico de boot y retorna si el sistema está listo.
    
    Uso típico en nexus.bat:
        python -c "import asyncio; from app.core.health_sentinel import run_boot_diagnostic; print(asyncio.run(run_boot_diagnostic()))"
    
    Returns:
        True si el sistema está listo para trading
    """
    report = await health_sentinel.run_full_diagnostic(health_url)
    
    if print_report:
        health_sentinel.print_health_card(report)
    
    if not report.is_ready_for_trading:
        logger.critical("[SENTINEL] 🛑 BOOT VALIDATION FAILED - System not ready for trading")
    
    return report.is_ready_for_trading


def run_diagnostic_sync(health_url: str = DEFAULT_HEALTH_ENDPOINT) -> bool:
    """Versión síncrona de run_boot_diagnostic."""
    return asyncio.run(run_boot_diagnostic(health_url))
