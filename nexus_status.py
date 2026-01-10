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
from datetime import datetime, timedelta
import urllib.request
import urllib.error

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def ok(msg): return f"{GREEN}[OK] {RESET} {msg}"
def err(msg): return f"{RED}[ERR]{RESET} {msg}"
def warn(msg): return f"{YELLOW}[⚠] {RESET} {msg}"
def info(msg): return f"{CYAN}[INF]{RESET} {msg}"


def check_zmq_bridge():
    """Check if ZMQ port 5555 is in use (bridge active)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 5555))
        sock.close()
        
        if result == 0:
            return ok("ZMQ Bridge ...... Puerto 5555 Activo")
        else:
            return warn("ZMQ Bridge ...... Puerto 5555 No responde (esperando MT5)")
    except Exception as e:
        return err(f"ZMQ Bridge ...... Error: {e}")


def check_rag_memory():
    """Check ChromaDB connection and vector count."""
    try:
        import chromadb
        from chromadb.config import Settings
        
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "/app/data/chroma")
        
        if not os.path.exists(persist_dir):
            return warn("RAG Memory ...... Directorio no existe (primera ejecución)")
            
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        
        total_vectors = 0
        for col in collections:
            total_vectors += col.count()
            
        return ok(f"RAG Memory ...... {total_vectors} Recuerdos indexados")
    except ImportError:
        return warn("RAG Memory ...... ChromaDB no instalado")
    except Exception as e:
        return err(f"RAG Memory ...... Error: {str(e)[:50]}")


def check_data_feed():
    """Check SQLite database and last tick timestamp."""
    db_path = "/app/data/market_data.db"
    
    if not os.path.exists(db_path):
        return warn("Data Feed ....... Base de datos no creada (sin datos aún)")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check total ticks
        cursor.execute("SELECT COUNT(*) FROM ticks")
        total_ticks = cursor.fetchone()[0]
        
        if total_ticks == 0:
            return warn("Data Feed ....... 0 ticks (esperando datos de MT5)")
        
        # Check last timestamp
        cursor.execute("SELECT MAX(timestamp) FROM ticks")
        last_ts = cursor.fetchone()[0]
        conn.close()
        
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", ""))
                age_seconds = (datetime.utcnow() - last_dt).total_seconds()
                
                if age_seconds < 60:
                    return ok(f"Data Feed ....... {total_ticks} ticks (último: hace {int(age_seconds)}s)")
                elif age_seconds < 300:
                    return warn(f"Data Feed ....... {total_ticks} ticks (último: hace {int(age_seconds)}s - LAG?)")
                else:
                    return err(f"Data Feed ....... Último dato hace {int(age_seconds)}s (MT5 Desconectado?)")
            except:
                return ok(f"Data Feed ....... {total_ticks} ticks almacenados")
        
        return ok(f"Data Feed ....... {total_ticks} ticks almacenados")
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return warn("Data Feed ....... Tabla ticks no existe (iniciando recolección)")
        return err(f"Data Feed ....... SQL Error: {e}")
    except Exception as e:
        return err(f"Data Feed ....... Error: {str(e)[:50]}")


def check_ml_models():
    """Check if ML models exist."""
    models_dir = "/app/models"
    gbm_path = os.path.join(models_dir, "gbm_v1.joblib")
    lstm_path = os.path.join(models_dir, "lstm_v1.pt")
    
    gbm_exists = os.path.exists(gbm_path)
    lstm_exists = os.path.exists(lstm_path)
    
    if gbm_exists and lstm_exists:
        return ok("ML Engine ....... Modelos GBM + LSTM cargados")
    elif gbm_exists:
        return warn("ML Engine ....... Solo GBM (LSTM pendiente)")
    elif lstm_exists:
        return warn("ML Engine ....... Solo LSTM (GBM pendiente)")
    else:
        return warn("ML Engine ....... Sin modelos (Shadow Mode - recolectando datos)")


def check_execution_mode():
    """Check if execution is enabled."""
    exec_enabled = os.environ.get("EXECUTION_ENABLED", "false").lower() == "true"
    
    if exec_enabled:
        return err("Execution Mode .. ⚠️ LIVE - Ejecución REAL activa")
    else:
        return ok("Execution Mode .. Shadow Mode (solo predicciones)")


def check_sse_api():
    """Check if SSE API is responding."""
    try:
        req = urllib.request.Request("http://localhost:8000/", method="GET")
        req.add_header("Accept", "text/event-stream")
        
        with urllib.request.urlopen(req, timeout=3) as response:
            return ok("API Server ...... Puerto 8000 SSE Activo")
    except urllib.error.URLError as e:
        return warn(f"API Server ...... No responde: {str(e.reason)[:30]}")
    except Exception as e:
        return warn(f"API Server ...... Error: {str(e)[:30]}")


def check_memory_usage():
    """Check container memory usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        used_mb = mem.used / (1024 * 1024)
        total_mb = mem.total / (1024 * 1024)
        percent = mem.percent
        
        if percent < 70:
            return ok(f"Memory Usage .... {used_mb:.0f}MB / {total_mb:.0f}MB ({percent}%)")
        elif percent < 85:
            return warn(f"Memory Usage .... {used_mb:.0f}MB / {total_mb:.0f}MB ({percent}%)")
        else:
            return err(f"Memory Usage .... {used_mb:.0f}MB / {total_mb:.0f}MB ({percent}%) ⚠️ ALTO")
    except ImportError:
        return info("Memory Usage .... psutil no disponible")
    except Exception as e:
        return info(f"Memory Usage .... No disponible")


def main():
    print()
    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║           FEAT SNIPER NEXUS - SYSTEM STATUS              ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════╝{RESET}")
    print()
    print(f"  Timestamp: {datetime.utcnow().isoformat()}Z")
    print()
    print(f"{BOLD}  ─── Core Systems ───{RESET}")
    print(f"  {check_zmq_bridge()}")
    print(f"  {check_sse_api()}")
    print()
    print(f"{BOLD}  ─── Data Layer ───{RESET}")
    print(f"  {check_data_feed()}")
    print(f"  {check_rag_memory()}")
    print()
    print(f"{BOLD}  ─── ML Engine ───{RESET}")
    print(f"  {check_ml_models()}")
    print(f"  {check_execution_mode()}")
    print()
    print(f"{BOLD}  ─── Resources ───{RESET}")
    print(f"  {check_memory_usage()}")
    print()
    print(f"{CYAN}  ────────────────────────────────────────────────────────{RESET}")
    print(f"  {GREEN}✓{RESET} Diagnóstico completado")
    print()


if __name__ == "__main__":
    main()
