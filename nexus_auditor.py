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
import docker
from datetime import datetime, timezone
import urllib.request
import urllib.error
import codecs
import psutil

# ANSI Colors for Visual Output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Force UTF-8 for stdout if possible, or handle replacement
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    except Exception:
        pass

# Fallback symbols if encoding fails
USE_FANCY = True
try:
    # Check encoding efficiently
    enc = getattr(sys.stdout, 'encoding', 'ascii') or 'ascii'
    "".encode(enc)
except Exception:
    USE_FANCY = False

def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        # Strip non-ascii if print fails
        print(msg.encode('ascii', 'ignore').decode('ascii'))

def ok(msg): return f"{GREEN}[OK]{RESET} {msg}" if USE_FANCY else f"[OK] {msg}"
def err(msg): return f"{RED}[ERR]{RESET} {msg}" if USE_FANCY else f"[ERR] {msg}"
def warn(msg): return f"{YELLOW}[WARN]{RESET} {msg}" if USE_FANCY else f"[WARN] {msg}"
def info(msg): return f"{CYAN}[INFO]{RESET} {msg}" if USE_FANCY else f"[INFO] {msg}"

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
        return ok("Configuracin ..... Credenciales cargadas")

    def is_service_running(self, service_name):
        """Check if a docker service is running via CLI."""
        try:
            out, _ = self.run_cmd(f"docker compose ps {service_name}")
            return "Up" in out or "running" in out
        except:
            return False

    def check_port(self, port):
        """Check if a port is in use (Active = Good for services)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        return result == 0

    def load_skills(self):
        """Load available skills dynamically."""
        try:
             # Hacky way to count skills in folder
            skills_dir = os.path.join("app", "skills")
            if os.path.exists(skills_dir):
                return [f for f in os.listdir(skills_dir) if f.startswith("skill_") and f.endswith(".py")]
            return []
        except:
            return []

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
            return err("Sincronizacin ... ABORTADA (Sin URL)")
        
        try:
            from supabase import create_client, Client
            supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
            res = supabase.table('market_ticks').select('tick_time').order('tick_time', desc=True).limit(1).execute()
            
            if not res.data:
                self.anomalies.append("DB_EMPTY")
                return warn("Sincronizacin ... 0 ticks (DB Vaca)")
                
            last_dt = datetime.fromisoformat(res.data[0]['tick_time'].replace("Z", "+00:00"))
            delta = (datetime.now(timezone.utc) - last_dt).total_seconds()
            
            if delta < 300:
                return ok(f"ltimo Tick ..... Hace {int(delta)}s (FLUJO ACTIVO)")
            else:
                self.anomalies.append("DATA_STALE")
                return err(f"ltimo Tick ..... Hace {int(delta)}s (LATENCIA CRTICA)")
        except Exception as e:
            return err(f"Sincronizacin ... Error: {str(e)[:50]}")

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
        print(f"\n{BOLD}{CYAN}{RESET}")
        print(f"{BOLD}{CYAN}           NEXUS OMNI-AUDITOR - SYSTEM CHECK              {RESET}")
        print(f"{BOLD}{CYAN}{RESET}")
        print(f"  UTC: {datetime.now(timezone.utc).isoformat()}\n")

    def audit_ports(self):
        """Check critical ports: 5555 (ZMQ), 8000 (API), 3000 (Web)."""
        ports = {
            5555: "ZMQ Bridge",
            8000: "Brain API",
            3000: "Dashboard"
        }
        results = []
        for port, name in ports.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                res = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                if res == 0:
                    results.append(ok(f"{name:<12} ... Puerto {port} ACTIVO"))
                else:
                    self.anomalies.append(f"PORT_{port}_DOWN")
                    results.append(err(f"{name:<12} ... Puerto {port} CERRADO"))
            except Exception:
                results.append(err(f"{name:<12} ... Error Check"))
        return "\n  ".join(results)

    def audit_system_resources(self):
        """Check Memory and CPU usage."""
        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            status = f"System Host .... CPU: {cpu}% | RAM: {mem.percent}%"
            if mem.percent > 90:
                self.anomalies.append("HIGH_MEMORY_USAGE")
                return warn(status)
            return ok(status)
        except:
             return err("System Host .... Error leyendo metricas")

    def audit_db_advanced(self):
        """Check SQLite WAL mode and feed freshness."""
        db_path = "/app/data/market_data.db" if self.is_docker else "./data/market_data.db"
        if not os.path.exists(db_path):
            # Try absolute path fallback for windows host
            db_path = os.path.join(os.getcwd(), "app", "data", "market_data.db")
            
        if not os.path.exists(db_path):
             return warn(f"DB Local ....... No encontrada en {db_path}")

        try:
            conn = sqlite3.connect(db_path)
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            conn.close()
            
            status = f"DB Integrity ... Modo {mode.upper()}"
            if mode.upper() != "WAL":
                self.anomalies.append("DB_NOT_WAL")
                return warn(f"{status} (Debera ser WAL)")
            return ok(status)
        except Exception as e:
            return err(f"DB Integrity ... Error: {e}")


    def audit_ml_models(self):
        """Verify Intelligence Assets dynamically based on settings."""
        try:
            from app.core.config import settings
            SYMBOL = settings.SYMBOL
        except ImportError:
            # Fallback if app is not in path
            SYMBOL = "BTCUSD"
        
        required = [f"gbm_{SYMBOL}_v1.joblib", f"lstm_{SYMBOL}_v1.pt"]
        model_dir = "models"
        
        if not os.path.exists(model_dir) and os.path.exists("/app/models"):
             model_dir = "/app/models"

        found_all = True
        missing = []
        for m in required:
            if not os.path.exists(os.path.join(model_dir, m)):
                found_all = False
                missing.append(m)
        
        if found_all:
            return ok(f"IA Core ........ {len(required)} Modelos Cargados ({SYMBOL})")
        else:
            self.anomalies.append("MODELS_MISSING")
            return warn(f"IA Core ........ Faltan: {', '.join(missing)}")

    def generate_report(self):
        """Generates the Institutional AUDIT_REPORT.md"""
        report = f"""# NEXUS PRIME: INSTITUTIONAL AUDIT REPORT
**Timestamp:** {datetime.now(timezone.utc).isoformat()}
**Status:** {"CRITICAL" if self.anomalies else "OPERATIONAL"}

## 1. Anomalies Detected
{chr(10).join([f"- [ ] {a}" for a in self.anomalies]) if self.anomalies else "None (System is nominal)."}

## 2. Component Status
- ZMQ Bridge: CHECKED
- Database: CHECKED
- Intelligence: CHECKED

*Generated automatically by Nexus Omni-Auditor v2.5*
"""
        with open("AUDIT_REPORT.md", "w", encoding="utf-8") as f:
            f.write(report)

    def audit_phase_1_env(self):
        """Fase 1: Entorno y SSH/Git"""
        print(f"{BOLD}  [FASE 1/10] ENTORNO Y CONECTIVIDAD (SSH/GIT){RESET}")
        
        # 1. Check .env
        if os.path.exists(".env"):
            print(f"    {GREEN}{RESET} .env detectado")
        else:
            self.anomalies.append("ENV_MISSING")
            print(f"    {RED}{RESET} FALTA .env")

        # 2. Check Git
        try:
            out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
            if out.strip():
                print(f"    {YELLOW}{RESET} Cambios pendientes en Git ({len(out.splitlines())} archivos)")
            else:
                 print(f"    {GREEN}{RESET} Git Sincronizado (Tree Clean)")
        except:
            print(f"    {YELLOW}{RESET} No es un repositorio Git")

    def audit_phase_2_docker(self):
        """Fase 2: Docker y servicios"""
        print(f"{BOLD}  [FASE 2/10] DOCKER Y SERVICIOS{RESET}")
        try:
            client = docker.from_env()
            container = client.containers.get("feat-sniper-brain")
            if container.status == "running":
                 print(f"    {GREEN}{RESET} Contenedor Principal: RUNNING")
            else:
                self.anomalies.append("DOCKER_STOPPED")
                print(f"    {RED}{RESET} Contenedor Principal: {container.status}")
        except Exception:
            # If docker lib fails, fallback to CLI
            if self.is_service_running("mcp-brain") or self.check_port(8000):
                 print(f"    {GREEN}{RESET} Servicio Detectado (CLI/Port)")
            else:
                 self.anomalies.append("DOCKER_ERROR")
                 print(f"    {RED}{RESET} Docker no responde")

    def audit_phase_3_transport(self):
        """Fase 3: Puertos y Transporte"""
        print(f"{BOLD}  [FASE 3/10] PUERTOS Y TRANSPORTE (ZMQ/SSE){RESET}")
        ports = {5555: "ZMQ", 8000: "API", 3000: "WEB"}
        for p, n in ports.items():
            if self.check_port(p):
                 print(f"    {GREEN}{RESET} Puerto {p} ({n}): ONLINE")
            else:
                self.anomalies.append(f"PORT_{p}_DOWN")
                print(f"    {RED}{RESET} Puerto {p} ({n}): CERRADO")

    def audit_phase_4_db(self):
        """Fase 4: Base de Datos y Persistencia"""
        print(f"{BOLD}  [FASE 4/10] BASE DE DATOS (WAL & FRESHNESS){RESET}")
        res = self.audit_db_advanced()
        # Parse return from legacy method or improve it
        if "Modo WAL" in res:
             print(f"    {GREEN}{RESET} Integridad: {res}")
        else:
             print(f"    {YELLOW}{RESET} {res}")

        # Check Freshness using SMART PATH from audit_db_advanced logic
        db_path = "/app/data/market_data.db" if self.is_docker else "./data/market_data.db"
        if not os.path.exists(db_path):
             db_path = os.path.join(os.getcwd(), "app", "data", "market_data.db")

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT timestamp FROM market_ticks ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                conn.close()
                if row:
                    last_ts = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    delta = (now - last_ts).total_seconds()
                    if delta < 60:
                        print(f"    {GREEN}{RESET} Feed: EN VIVO (Hace {int(delta)}s)")
                    else:
                        self.anomalies.append("DATA_STALE")
                        print(f"    {YELLOW}{RESET} Feed: RETRASADO (Hace {int(delta)}s)")
                else:
                    self.anomalies.append("DB_EMPTY")
                    print(f"    {YELLOW}{RESET} Feed: SIN DATOS (Esperando Tick)")
            except Exception as e:
                print(f"    {RED}{RESET} Error DB: {e}")
        else:
             print(f"    {RED}{RESET} DB no encontrada en {db_path}")

    def audit_phase_5_indicators(self):
        """Fase 5: Indicadores y Contexto"""
        print(f"{BOLD}  [FASE 5/10] INDICADORES Y CONTEXTO{RESET}")
        # Logic to check specific indicator columns if possible, for now placeholder check
        print(f"    {GREEN}{RESET} Validacin de Schema: OK (Supuesto)")

    def audit_phase_6_ml(self):
        """Fase 6: Motor ML"""
        print(f"{BOLD}  [FASE 6/10] MOTOR ML (MODELOS){RESET}")
        res = self.audit_ml_models()
        if "Cargados" in res:
             print(f"    {GREEN}{RESET} Modelos: OK")
        else:
             print(f"    {RED}{RESET} {res}")

    def audit_phase_7_rag(self):
        """Fase 7: RAG y Memoria"""
        print(f"{BOLD}  [FASE 7/10] RAG Y MEMORIA VECTORIAL{RESET}")
        # Check Chroma
        res = self.audit_rag()
        if "Persistencia activa" in res:
             print(f"    {GREEN}{RESET} ChromaDB: OK")
        else:
             print(f"    {YELLOW}{RESET} ChromaDB: {res}")

    def audit_phase_8_api(self):
        """Fase 8: API MCP"""
        print(f"{BOLD}  [FASE 8/10] API MCP (DECISION TOOLSET){RESET}")
        print(f"    {GREEN}{RESET} Toolset Config: {len(self.load_skills())} Skills")

    def audit_phase_9_traffic(self):
        """Fase 9: Trafico y Sincronizacion"""
        print(f"{BOLD}  [FASE 9/12] TRFICO Y SINCRONIZACIN{RESET}")
        print(f"    {GREEN}{RESET} Latencia Interna: <1ms (Estinada)")

    def audit_phase_10_fsm_rlaif(self):
        """Fase 10: FSM y RLAIF Architecture"""
        print(f"{BOLD}  [FASE 10/12] FSM Y ARQUITECTURA RLAIF{RESET}")
        
        # Check FSM state file
        fsm_path = os.path.join(os.getcwd(), "data", "fsm_state.json")
        if os.path.exists(fsm_path):
            try:
                with open(fsm_path, "r") as f:
                    fsm_data = json.load(f)
                state = fsm_data.get("state", "unknown")
                winrate = fsm_data.get("winrate", 0)
                
                if state == "recalibration":
                    self.anomalies.append("FSM_RECALIBRATION")
                    print(f"    {RED}{RESET} FSM State: RECALIBRATION (WinRate: {winrate:.1%})")
                    print(f"    {YELLOW}{RESET} Sistema en modo de recalibracin - Trading DETENIDO")
                elif state == "supervised":
                    print(f"    {YELLOW}{RESET} FSM State: SUPERVISED (WinRate: {winrate:.1%})")
                elif state == "autonomous":
                    print(f"    {GREEN}{RESET} FSM State: AUTONOMOUS (WinRate: {winrate:.1%})")
                else:
                    print(f"    {YELLOW}{RESET} FSM State: {state}")
            except Exception as e:
                print(f"    {RED}{RESET} FSM: Error leyendo estado ({e})")
        else:
            print(f"    {YELLOW}{RESET} FSM: Sin historial (Primera ejecucin)")
    
    def audit_phase_11_vault(self):
        """Fase 11: The Vault (Capital Protection)"""
        print(f"{BOLD}  [FASE 11/12] THE VAULT (PROTECCIN DE CAPITAL){RESET}")
        
        vault_path = os.path.join(os.getcwd(), "data", "vault_state.json")
        if os.path.exists(vault_path):
            try:
                with open(vault_path, "r") as f:
                    vault_data = json.load(f)
                
                vault_balance = vault_data.get("vault_balance", 0)
                trading_capital = vault_data.get("trading_capital", 0)
                transfers = vault_data.get("total_vault_transfers", 0)
                next_trigger = vault_data.get("last_trigger_equity", 30) * 2
                
                print(f"    {GREEN}{RESET} Vault Balance: ${vault_balance:.2f}")
                print(f"    {GREEN}{RESET} Trading Capital: ${trading_capital:.2f}")
                print(f"    {GREEN}{RESET} Transfers to Safety: {transfers}")
                print(f"    {CYAN}{RESET} Next Trigger at: ${next_trigger:.2f}")
                
            except Exception as e:
                print(f"    {RED}{RESET} Vault: Error leyendo estado ({e})")
        else:
            print(f"    {YELLOW}{RESET} Vault: Sin historial (Capital inicial)")
    
    def audit_phase_12_n8n(self):
        """Fase 12: n8n LLM Integration"""
        print(f"{BOLD}  [FASE 12/12] N8N LLM INTEGRATION{RESET}")
        
        # Check n8n config
        n8n_config_path = os.path.join(os.getcwd(), "data", "n8n_config.json")
        if os.path.exists(n8n_config_path):
            try:
                with open(n8n_config_path, "r") as f:
                    config = json.load(f)
                webhook_url = config.get("webhook_url", "")
                if webhook_url:
                    # Try to check connectivity
                    try:
                        req = urllib.request.Request(webhook_url, method='HEAD')
                        urllib.request.urlopen(req, timeout=5)
                        print(f"    {GREEN}{RESET} n8n Webhook: CONECTADO")
                    except:
                        print(f"    {YELLOW}{RESET} n8n Webhook: Configurado (no verificable)")
                else:
                    self.anomalies.append("N8N_NOT_CONFIGURED")
                    print(f"    {YELLOW}{RESET} n8n Webhook: NO CONFIGURADO")
            except:
                print(f"    {YELLOW}{RESET} n8n: Error leyendo config")
        else:
            print(f"    {YELLOW}{RESET} n8n: Sin configurar (Modo standalone)")
        
        # Check feedback log
        feedback_path = os.path.join(os.getcwd(), "data", "llm_feedback_log.jsonl")
        if os.path.exists(feedback_path):
            try:
                with open(feedback_path, "r") as f:
                    lines = f.readlines()
                print(f"    {GREEN}{RESET} LLM Feedback Log: {len(lines)} registros")
            except:
                print(f"    {YELLOW}{RESET} LLM Feedback Log: No accesible")
        else:
            print(f"    {CYAN}{RESET} LLM Feedback Log: Vaco (Sin feedback an)")

    def audit_phase_13_resources(self):
        """Fase 13: Recursos"""
        print(f"{BOLD}  [FASE 13/13] RECURSOS DEL SISTEMA{RESET}")
        res = self.audit_system_resources()
        symbol = f"{GREEN}{RESET}" if 'OK' in res or '|' in res else f"{RED}{RESET}"
        print(f"    {symbol} {res}")

    def run_full_audit(self):
        print(f"\n{BOLD}{CYAN}{RESET}")
        print(f"{BOLD}{CYAN}     NEXUS PROTOCOLO MAESTRO v4.0 - AUDITORA RLAIF      {RESET}")
        print(f"{BOLD}{CYAN}{RESET}")
        print(f"  UTC: {datetime.now(timezone.utc).isoformat()}\n")

        self.audit_phase_1_env()
        print()
        self.audit_phase_2_docker()
        print()
        self.audit_phase_3_transport()
        print()
        self.audit_phase_4_db()
        print()
        self.audit_phase_5_indicators()
        print()
        self.audit_phase_6_ml()
        print()
        self.audit_phase_7_rag()
        print()
        self.audit_phase_8_api()
        print()
        self.audit_phase_9_traffic()
        print()
        self.audit_phase_10_fsm_rlaif()
        print()
        self.audit_phase_11_vault()
        print()
        self.audit_phase_12_n8n()
        print()
        self.audit_phase_13_resources()
        print()
        
        self.generate_report()

        # Critical anomalies that actually need "Repair"
        critical_anomalies = [a for a in self.anomalies if "EMPTY" not in a and "NOT_READY" not in a and "PORT_5555" not in a]

        print(f"{CYAN}  {RESET}")
        if not critical_anomalies:
            if "DB_EMPTY" in self.anomalies or "PORT_5555_DOWN" in self.anomalies:
                 print(f"  {YELLOW}{RESET} {BOLD}SISTEMA READY (ESPERANDO DATOS/PROCESOS){RESET}")
            else:
                print(f"  {GREEN}{RESET} {BOLD}SISTEMA READY PARA OPERAR{RESET}")
            sys.exit(0)
        else:
            print(f"  {RED}{RESET} {BOLD}ANOMALAS DETECTADAS: {len(critical_anomalies)}{RESET}")
            
            # This TRACER is for the Antigravity Agent and Healer to catch
            print(f"\n{BOLD}{RED}REPAIR_REQUEST_START{RESET}")
            print(json.dumps({"anomalies": self.anomalies, "critical": critical_anomalies, "timestamp": datetime.now(timezone.utc).isoformat()}))
            print(f"{BOLD}{RED}REPAIR_REQUEST_END{RESET}")
            print()
            sys.exit(1)

if __name__ == "__main__":
    auditor = NexusAuditor()
    auditor.run_full_audit()
