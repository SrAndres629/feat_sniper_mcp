#!/usr/bin/env python3
"""
NEXUS OMNI-AUDITOR - The Single Source of Truth
===============================================
Consolidated diagnostic engine for FEAT Sniper NEXUS.
Checks: Config, Connectivity, Processes, Intelligence, and MCP Skills.

Outputs "NEXUS_REPAIR_NEEDED" if anomalies are found to trigger Agent auto-healing.
"""

import os
import sys
import socket
import sqlite3
import glob
import subprocess
import json
from datetime import datetime, timezone
import urllib.request
import urllib.error

# ANSI Colors for Visual Output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"
BOLD = "\033[1m"

def ok(msg): return f"{GREEN}[OK] {RESET} {msg}"
def err(msg): return f"{RED}[ERR]{RESET} {msg}"
def warn(msg): return f"{YELLOW}[⚠] {RESET} {msg}"
def info(msg): return f"{CYAN}[INF]{RESET} {msg}"

def load_env():
    """Loads .env into os.environ for local execution."""
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    parts = line.strip().split("=", 1)
                    if len(parts) == 2:
                        k, v = parts
                        if k not in os.environ:
                            os.environ[k] = v
load_env()

class NexusAuditor:
    def __init__(self):
        self.anomalies = []
        # Robust check to see if we are REALLY inside a container
        self.is_docker = os.path.exists('/.dockerenv')

    def audit_config(self):
        """Verify essential environment variables."""
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        missing = []
        if not url: missing.append("SUPABASE_URL")
        if not key: missing.append("SUPABASE_KEY")
        
        if missing:
            msg = f"Variables faltantes: {', '.join(missing)}"
            self.anomalies.append(f"CONFIG_MISSING_{'_'.join(missing)}")
            return err(msg)
        return ok("Configuración ..... Credenciales cargadas")

    def audit_zmq(self):
        """Check ZMQ Bridge status."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            # Check local port 5555
            result = sock.connect_ex(('127.0.0.1', 5555))
            sock.close()
            if result == 0:
                return ok("ZMQ Bridge ...... Puerto 5555 Activo")
            else:
                self.anomalies.append("ZMQ_NOT_READY")
                return warn("ZMQ Bridge ...... No responde (esperando MT5)")
        except Exception as e:
            return err(f"ZMQ Bridge ...... Error: {e}")

    def audit_mt5(self):
        """Check if MT5 is running (Windows Host only) using psutil."""
        if os.name != 'nt':
            return info("MT5 Terminal .... Saltado (Modo Docker)")
        
        try:
            import psutil
            found = False
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and 'terminal64.exe' in proc.info['name'].lower():
                    found = True
                    break
            
            if found:
                return ok("MT5 Terminal .... CORRIENDO (VIVO)")
            else:
                self.anomalies.append("MT5_OFFLINE")
                return err("MT5 Terminal .... NO DETECTADO")
        except Exception as e:
            return warn(f"MT5 Terminal .... Error al verificar: {e}")

    def audit_supabase(self):
        """Check live connection to Supabase and data flow."""
        if "SUPABASE_URL" not in os.environ:
            return err("Sincronización ... ABORTADA (Sin URL)")
        
        try:
            from supabase import create_client, Client
            supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
            res = supabase.table('market_ticks').select('tick_time').order('tick_time', desc=True).limit(1).execute()
            
            if not res.data:
                self.anomalies.append("DB_EMPTY")
                return warn("Sincronización ... 0 ticks (DB Vacía)")
                
            last_dt = datetime.fromisoformat(res.data[0]['tick_time'].replace("Z", "+00:00"))
            delta = (datetime.now(timezone.utc) - last_dt).total_seconds()
            
            if delta < 300:
                return ok(f"Último Tick ..... Hace {int(delta)}s (FLUJO ACTIVO)")
            else:
                self.anomalies.append("DATA_STALE")
                return err(f"Último Tick ..... Hace {int(delta)}s (LATENCIA CRÍTICA)")
        except Exception as e:
            return err(f"Sincronización ... Error: {str(e)[:50]}")

    def audit_rag(self):
        """Check ChromaDB state (Smart Path Detection)."""
        try:
            # Try Docker path first, then local fallback
            p_docker = os.environ.get("CHROMA_PERSIST_DIR", "/app/data/chroma")
            p_local = "./chroma_storage" if not self.is_docker else p_docker
            
            path = p_local if os.path.exists(p_local) else p_docker
            
            if not os.path.exists(path):
                self.anomalies.append("RAG_MISSING")
                return warn(f"RAG Memory ...... No encontrado en {path}")
            
            files = os.listdir(path)
            return ok(f"RAG Memory ...... Persistencia activa ({len(files)} items)")
        except Exception:
            return err("RAG Memory ...... Error de acceso")

    def audit_mcp_skills(self):
        """Verify MQL5 skills registration in MCP Server."""
        if self.is_docker:
            return info("MCP Skills ...... Saltado (En contenedor)")
            
        try:
            sys.path.append(os.getcwd())
            from mcp_server import mcp
            import asyncio
            
            async def get_tools(): return await mcp.get_tools()
            tools = asyncio.run(get_tools())
            names = [getattr(t, "name", str(t)) for t in tools]
            skills = [n for n in names if "skill_" in n]
            
            if skills:
                return ok(f"MCP Skills ...... {len(skills)} detectadas")
            else:
                self.anomalies.append("NO_SKILLS")
                return warn("MCP Skills ...... 0 detectadas")
        except Exception as e:
            return warn(f"MCP Skills ...... Error: {e}")

    def run_full_audit(self):
        print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗{RESET}")
        print(f"{BOLD}{CYAN}║           NEXUS OMNI-AUDITOR - SYSTEM CHECK              ║{RESET}")
        print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════╝{RESET}")
        print(f"  UTC: {datetime.now(timezone.utc).isoformat()}\n")

        print(f"{BOLD}  [1] INFRAESTRUCTURA{RESET}")
        print(f"  {self.audit_config()}")
        print(f"  {self.audit_zmq()}")
        print(f"  {self.audit_mt5()}")
        print()

        print(f"{BOLD}  [2] PERSISTENCIA Y DATOS{RESET}")
        print(f"  {self.audit_supabase()}")
        print(f"  {self.audit_rag()}")
        print()

        print(f"{BOLD}  [3] INTELIGENCIA Y MCP{RESET}")
        print(f"  {self.audit_mcp_skills()}")
        print()

        print(f"{CYAN}  ────────────────────────────────────────────────────────{RESET}")
        if not self.anomalies:
            print(f"  {GREEN}✓{RESET} {BOLD}SISTEMA READY PARA OPERAR{RESET}")
        else:
            print(f"  {RED}✗{RESET} {BOLD}ANOMALÍAS DETECTADAS: {len(self.anomalies)}{RESET}")
            # This TRACER is for the Antigravity Agent to catch
            print(f"\n{BOLD}{RED}REPAIR_REQUEST_START{RESET}")
            print(json.dumps({"anomalies": self.anomalies, "timestamp": datetime.now(timezone.utc).isoformat()}))
            print(f"{BOLD}{RED}REPAIR_REQUEST_END{RESET}")
        print()

if __name__ == "__main__":
    auditor = NexusAuditor()
    auditor.run_full_audit()
