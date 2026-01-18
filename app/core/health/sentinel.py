import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict
from .foundation import HealthReport, HealthState, SubsystemState, SubsystemCheck, DEFAULT_HEALTH_ENDPOINT, LATENCY_WARNING_THRESHOLD_MS, request_with_retry, validate_health_schema
from .checks import check_ml_readiness, check_physics_engine, check_feat_gates

logger = logging.getLogger("Health.Sentinel")

class HealthSentinel:
    _instance: Optional['HealthSentinel'] = None
    
    def __new__(cls):
        if cls._instance is None: cls._instance = super().__new__(cls); cls._instance._init = False
        return cls._instance

    def __init__(self):
        if self._init: return
        self._init = True
        self._last_report = None
        logger.info("HealthSentinel initialized")

    async def check_endpoint_health(self, url: str) -> SubsystemCheck:
        d, lat, err = await request_with_retry(url)
        if err: return SubsystemCheck("MCP Server", SubsystemState.ERROR, err, latency_ms=lat)
        ok, warns = validate_health_schema(d)
        st = SubsystemState.READY if ok and not warns else (SubsystemState.DEGRADED if ok else SubsystemState.NOT_READY)
        msg = "Healthy" if st == SubsystemState.READY else ("; ".join(warns) if warns else "Schema invalid")
        return SubsystemCheck("MCP Server", st, msg, latency_ms=lat, metadata=d)

    async def run_full_diagnostic(self, url: str = DEFAULT_HEALTH_ENDPOINT) -> HealthReport:
        logger.info("Starting diagnostic...")
        subs = [
            await self.check_endpoint_health(url),
            await check_ml_readiness(),
            await check_physics_engine(),
            await check_feat_gates()
        ]
        
        errs = sum(1 for s in subs if s.state == SubsystemState.ERROR)
        nr = sum(1 for s in subs if s.state == SubsystemState.NOT_READY)
        deg = sum(1 for s in subs if s.state == SubsystemState.DEGRADED)
        
        ov = HealthState.HEALTHY
        if errs > 0: ov = HealthState.UNHEALTHY
        elif nr > 0 or deg > 1: ov = HealthState.DEGRADED
        
        warns = []
        if subs[0].latency_ms and subs[0].latency_ms > LATENCY_WARNING_THRESHOLD_MS:
            warns.append(f"High latency: {subs[0].latency_ms:.1f}ms")
            
        r = HealthReport(ov, datetime.now().isoformat(), subs, subs[0].latency_ms or 0, warns, ov == HealthState.HEALTHY)
        self._last_report = r
        return r

    def print_health_card(self, r: Optional[HealthReport] = None):
        r = r or self._last_report
        if not r: return print("No report")
        sym = {SubsystemState.READY: "✅", SubsystemState.WARMING_UP: "⏳", SubsystemState.DEGRADED: "⚠️", SubsystemState.NOT_READY: "❌", SubsystemState.ERROR: "💀"}
        ov_sym = {HealthState.HEALTHY: "🟢 HEALTHY", HealthState.DEGRADED: "🟡 DEGRADED", HealthState.UNHEALTHY: "🔴 UNHEALTHY", HealthState.UNKNOWN: "⚪ UNKNOWN"}
        print(f"\n{'='*60}\n  FEAT SNIPER - Health Report\n{'='*60}")
        print(f"  Status: {ov_sym.get(r.overall_state)}\n  Trading OK: {'YES ✅' if r.is_ready_for_trading else 'NO ❌'}\n{'-'*60}")
        for s in r.subsystems: print(f"    {sym.get(s.state,'?')} {s.name}: {s.message} ({s.latency_ms or 0:.1f}ms)")
        if r.warnings: 
            print(f"{'-'*60}\n  Warnings:")
            for w in r.warnings: print(f"    ⚠️ {w}")
        print(f"{'='*60}\n")
    
    def get_status(self) -> Dict:
        if not self._last_report: return {"has_report": False}
        r = self._last_report
        return {"has_report": True, "state": r.overall_state.value, "ready": r.is_ready_for_trading}

health_sentinel = HealthSentinel()
