#!/usr/bin/env python3
"""
NEXUS STATUS - Diagnostic Tool
==============================
Visual scanner for FEAT Sniper NEXUS health.
Runs inside Docker container and checks all vital systems.

Uses only standard Python libraries (no rebuild required).
"""


import os
import sys
import socket
import sqlite3
import glob
from datetime import datetime, timezone
import urllib.request
import urllib.error
import subprocess

# Add .env loading support for local execution
def load_env():
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    if k not in os.environ:
                        os.environ[k] = v

load_env()

# ANSI Colors
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

def check_config_validity():
    """Verify if environment variables are set."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    missing = []
    if not url: missing.append("SUPABASE_URL")
    if not key: missing.append("SUPABASE_KEY")
    
    if missing:
        return err(f"Variables faltantes: {', '.join(missing)}")
    return ok("Configuración ..... Credenciales cargadas")

def check_zmq_bridge():
    """Check if ZMQ port 5555 is in use (bridge active)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        # In Docker, we check 127.0.0.1
        result = sock.connect_ex(('127.0.0.1', 5555))
        sock.close()
        
        if result == 0:
            return ok("ZMQ Bridge ...... Puerto 5555 Activo")
        else:
            return warn("ZMQ Bridge ...... Puerto 5555 No responde (esperando MT5)")
    except Exception as e:
        return err(f"ZMQ Bridge ...... Error: {e}")

def check_mt5_terminal():
    """Verify if MT5 is running on the host (Windows)."""
    if os.name != 'nt':
        return info("MT5 Terminal .... Saltado (Modo Docker/Linux)")
    
    try:
        # Using tasklist via subprocess since psutil might not be there
        output = subprocess.check_output('tasklist /FI "IMAGENAME eq terminal64.exe" /NH', shell=True).decode()
        if "terminal64.exe" in output:
            return ok("MT5 Terminal .... CORRIENDO (VIVO)")
        else:
            return err("MT5 Terminal .... NO DETECTADO (Cerrado?)")
    except Exception:
        return warn("MT5 Terminal .... No se pudo verificar (Permisos?)")

def check_data_freshness():
    """SRE-Grade Sync Check: Measures LAG against Supabase (Schema Fixed)."""
    config_status = check_config_validity()
    if "[ERR]" in config_status:
        return f"  {config_status}\n  {err('Sincronización ... ABORTADA p/ falta de config')}"

    try:
        from supabase import create_client, Client
    except ImportError:
        return warn("Sincronización ... Librería 'supabase' no instalada")
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
        
    try:
        supabase: Client = create_client(url, key)
        
        # SCHEMA FIX: Using 'tick_time' instead of 'timestamp'
        res = supabase.table('market_ticks').select('tick_time, symbol, bid').order('tick_time', desc=True).limit(1).execute()
        
        if not res.data:
            return warn("Sincronización ... 0 ticks encontrados (Base de Datos vacía)")

        last_tick = res.data[0]
        last_ts_str = last_tick['tick_time']
        symbol = last_tick.get('symbol', 'N/A')
        price = last_tick.get('bid', 0.0)

        # Parse UTC Timestamp
        last_dt = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
        now_utc = datetime.now(timezone.utc)
        
        delta_seconds = int((now_utc - last_dt).total_seconds())
        if delta_seconds < 0: delta_seconds = 0
        
        status_line = f"Último Tick ..... Hace {delta_seconds}s ({symbol} @ {price})"
        
        if delta_seconds < 60:
            return f"  {ok(status_line)}\n  {ok('Estado .......... FLUJO EN TIEMPO REAL')}"
        elif delta_seconds < 300:
            return f"  {warn(status_line)}\n  {warn('Estado .......... LATENCIA ALTA')}"
        else:
            return f"  {err(status_line)}\n  {err('Estado .......... DESCONECTADO (Stale Data)')}"
            
    except Exception as e:
        return err(f"Sincronización ... Error: {str(e)[:50]}")

def get_last_logs(lines=15):
    """Retrieve last lines of log if possible."""
    # Try common log locations or general output
    log_files = glob.glob("/app/*.log") + glob.glob("/app/data/*.log")
    if not log_files:
        # Fallback to system check if we can (though in docker it's tricky)
        return "No se encontraron archivos .log específicos en /app"
    
    # Get newest log
    latest_log = max(log_files, key=os.path.getmtime)
    try:
        with open(latest_log, 'r') as f:
            content = f.readlines()
            return "".join(content[-lines:])
    except Exception as e:
        return f"Error al leer log {latest_log}: {e}"

def check_process_health():
    """Verify if vital processes are running + Autopsia logic."""
    try:
        import psutil
        procs = []
        for p in psutil.process_iter(['cmdline']):
            try:
                cmd = " ".join(p.info['cmdline'] or [])
                if "data_collector.py" in cmd or "mcp_server.py" in cmd:
                    procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if procs:
            return ok(f"Procesos ........ VIVOS (Nucleus Active)")
        else:
            autopsy = get_last_logs()
            return f"{err('Procesos ........ data_collector.py MUERTO')}\n{WHITE}─────── AUTOPSIA: ÚLTIMO LOG ───────{RESET}\n{autopsy}\n{WHITE}────────────────────────────────────{RESET}"
    except ImportError:
        return info("Procesos ........ psutil no disponible")
    except Exception as e:
        return warn(f"Procesos ........ Error al verificar: {str(e)[:30]}")

def check_rag_memory():
    """Check ChromaDB volume."""
    try:
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "/app/data/chroma")
        if not os.path.exists(persist_dir):
            return warn("RAG Memory ...... Vacío (sin directorio)")
        files = os.listdir(persist_dir)
        return ok(f"RAG Memory ...... {len(files)} archivos de persistencia")
    except Exception:
        return warn("RAG Memory ...... Error al acceder")

    except Exception:
        return warn("API Server ...... Puerto 8000 No responde")

def check_mcp_tools():
    """Verify if MCP tools/skills are correctly registered (Local only)."""
    if os.environ.get("DOCKER_MODE") == "true":
        return info("MCP Tools ....... Saltado (Verificar en Host)")
    
    try:
        # Import mcp_server safely
        sys.path.append(os.getcwd())
        from mcp_server import mcp
        import asyncio
        
        async def get_tools():
            return await mcp.get_tools()
        
        tools = asyncio.run(get_tools())
        names = []
        for t in tools:
            if hasattr(t, "name"): names.append(t.name)
            elif isinstance(t, str): names.append(t)
            elif isinstance(t, dict): names.append(t.get("name"))
            
        skills = [n for n in names if n.startswith("skill_")]
        if skills:
            return ok(f"MCP Tools ....... {len(skills)} skills detectadas")
        else:
            return warn("MCP Tools ....... 0 skills detectadas (Check path)")
    except Exception as e:
        return warn(f"MCP Tools ....... No se pudo verificar: {e}")

def main():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║           FEAT SNIPER NEXUS - SRE DASHBOARD              ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════╝{RESET}")
    print(f"\n  {WHITE}Timestamp UTC: {datetime.now(timezone.utc).isoformat()}{RESET}\n")
    
    print(f"{BOLD}  ─── Configuración ───{RESET}")
    print(f"  {check_config_validity()}")
    print()

    print(f"{BOLD}  ─── Sincronización ───{RESET}")
    print(check_data_freshness())
    print()
    
    print(f"{BOLD}  ─── Core Health ───{RESET}")
    print(f"  {check_zmq_bridge()}")
    print(f"  {check_mt5_terminal()}")
    print(f"  {check_process_health()}")
    print(f"  {check_sse_api()}")
    print()
    
    print(f"  {BOLD}─── Intelligence ───{RESET}")
    print(f"  {check_rag_memory()}")
    print(f"  {check_mcp_tools()}")
    print()
    
    print(f"{CYAN}  ────────────────────────────────────────────────────────{RESET}")
    print(f"  {GREEN}✓{RESET} Auditoría NEXUS completada")
    print()

if __name__ == "__main__":
    main()

