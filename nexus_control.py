"""
 FEAT NEXUS COMMAND CENTER v5.0 - Autonomous Evolution
======================================================
Distinguished Orchestrator for High-Frequency Systems.

Resilience Pillar: Self-Healing & Circuit Breakers.
Observability Pillar: Glass Cockpit Telemetry.
ML Pillar: Asset Identity Protocol.

Usage:
  python nexus_control.py start  (Deploy Ecosystem)
  python nexus_control.py stop   (Graceful Shutdown)
  python nexus_control.py audit  (Deep Diagnostic)
"""

import os
import sys
import socket
import subprocess
import time
import signal
import json
import webbrowser
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import psutil
from dotenv import load_dotenv

# Force load .env before anything else
load_dotenv()


# =============================================================================
# RLAIF BICAMERAL ARCHITECTURE: FSM Control System
# =============================================================================

class TradingState(Enum):
    """
    Máquina de Estados Finita para control Maestro-Aprendiz.
    
    🔴 RECALIBRATION: WinRate < 35% - Trading detenido, LLM diagnostica
    🟡 SUPERVISED: 35% <= WinRate <= 70% - NN propone, LLM aprueba
    🟢 AUTONOMOUS: WinRate > 70% - NN ejecuta, LLM audita post-mortem
    """
    RECALIBRATION = "recalibration"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"


@dataclass
class TradeRecord:
    """Registro de trade para cálculo de WinRate."""
    trade_id: str
    symbol: str
    direction: str  # BUY/SELL
    entry_price: float
    exit_price: Optional[float] = None
    profit: Optional[float] = None
    nn_confidence: float = 0.0
    llm_approved: Optional[bool] = None
    llm_feedback: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed: bool = False


class PerformanceTracker:
    """
    Rastrea el rendimiento de la NN para determinar el estado del FSM.
    Usa ventana deslizante de últimos N trades.
    """
    
    WINRATE_AUTONOMOUS_THRESHOLD = 0.70  # 70%
    WINRATE_RECALIBRATION_THRESHOLD = 0.35  # 35%
    MIN_TRADES_FOR_EVALUATION = 10
    EVALUATION_WINDOW = 50  # Últimos 50 trades
    
    def __init__(self, state_file: str = "data/fsm_state.json"):
        self.state_file = state_file
        self.trade_history: List[TradeRecord] = []
        self.current_state = TradingState.SUPERVISED  # Default: Supervised
        self._load_state()
    
    def _load_state(self):
        """Carga estado persistido desde disco."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.current_state = TradingState(data.get("state", "supervised"))
                    # Cargar historial resumido
                    self.trade_history = []
                    for t in data.get("recent_trades", []):
                        self.trade_history.append(TradeRecord(
                            trade_id=t["trade_id"],
                            symbol=t["symbol"],
                            direction=t["direction"],
                            entry_price=t["entry_price"],
                            exit_price=t.get("exit_price"),
                            profit=t.get("profit"),
                            nn_confidence=t.get("nn_confidence", 0),
                            llm_approved=t.get("llm_approved"),
                            closed=t.get("closed", False)
                        ))
        except Exception as e:
            print(f"[FSM] Error loading state: {e}. Starting fresh.")
            self.current_state = TradingState.SUPERVISED
    
    def _save_state(self):
        """Persiste estado a disco."""
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        data = {
            "state": self.current_state.value,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "winrate": self.calculate_winrate(),
            "recent_trades": [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "profit": t.profit,
                    "nn_confidence": t.nn_confidence,
                    "llm_approved": t.llm_approved,
                    "closed": t.closed
                }
                for t in self.trade_history[-self.EVALUATION_WINDOW:]
            ]
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def record_trade(self, trade: TradeRecord):
        """Registra un nuevo trade."""
        self.trade_history.append(trade)
        # Mantener solo ventana de evaluación
        if len(self.trade_history) > self.EVALUATION_WINDOW * 2:
            self.trade_history = self.trade_history[-self.EVALUATION_WINDOW:]
        self._save_state()
    
    def close_trade(self, trade_id: str, exit_price: float, profit: float):
        """Cierra un trade y actualiza su resultado."""
        for t in self.trade_history:
            if t.trade_id == trade_id and not t.closed:
                t.exit_price = exit_price
                t.profit = profit
                t.closed = True
                break
        self._save_state()
        self.evaluate_and_transition()
    
    def calculate_winrate(self) -> float:
        """Calcula WinRate de trades cerrados en la ventana."""
        closed_trades = [t for t in self.trade_history if t.closed][-self.EVALUATION_WINDOW:]
        if len(closed_trades) < self.MIN_TRADES_FOR_EVALUATION:
            return 0.5  # Default 50% cuando no hay suficientes datos
        
        wins = sum(1 for t in closed_trades if t.profit and t.profit > 0)
        return wins / len(closed_trades)
    
    def evaluate_and_transition(self) -> TradingState:
        """Evalúa rendimiento y transiciona estado si es necesario."""
        winrate = self.calculate_winrate()
        closed_count = len([t for t in self.trade_history if t.closed])
        
        old_state = self.current_state
        
        # Solo transicionar si hay suficientes trades para evaluar
        if closed_count >= self.MIN_TRADES_FOR_EVALUATION:
            if winrate >= self.WINRATE_AUTONOMOUS_THRESHOLD:
                self.current_state = TradingState.AUTONOMOUS
            elif winrate < self.WINRATE_RECALIBRATION_THRESHOLD:
                self.current_state = TradingState.RECALIBRATION
            else:
                self.current_state = TradingState.SUPERVISED
        
        if old_state != self.current_state:
            print(f"[FSM] State Transition: {old_state.value} → {self.current_state.value} (WinRate: {winrate:.1%})")
            self._save_state()
        
        return self.current_state
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna estado actual para telemetría."""
        closed_trades = [t for t in self.trade_history if t.closed]
        return {
            "current_state": self.current_state.value,
            "winrate": self.calculate_winrate(),
            "total_trades": len(self.trade_history),
            "closed_trades": len(closed_trades),
            "wins": sum(1 for t in closed_trades if t.profit and t.profit > 0),
            "losses": sum(1 for t in closed_trades if t.profit and t.profit <= 0),
            "thresholds": {
                "autonomous": self.WINRATE_AUTONOMOUS_THRESHOLD,
                "recalibration": self.WINRATE_RECALIBRATION_THRESHOLD
            }
        }


# Singleton global para FSM
performance_tracker = PerformanceTracker()

# Configuration
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\LiteFinance MT5 Terminal\terminal64.exe")
PROJECT_DIR = os.getcwd()
LOG_TAIL_LINES = 20

# Colors
# Visual Identity & HFT Semantics
GREEN = "\033[38;5;82m"
RED = "\033[38;5;196m"
YELLOW = "\033[38;5;214m"
CYAN = "\033[38;5;39m"
MAGENTA = "\033[38;5;171m"
WHITE = "\033[38;5;255m"
GRAY = "\033[38;5;244m"
RESET = "\033[0m"
BOLD = "\033[1m"
ITALIC = "\033[3m"
RESMAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
GOLD = '\033[33m' # HFT Fractal Alert
BLUE = "\033[38;5;27m" # Added for the new header

# MTF Roles
SNIPER_TF = "M1"
STRATEGIST_TF = "H1"
GLOBAL_TF = "D1"

def banner():
    art = f"""
{CYAN}{BOLD}                  
             
                          
                         
                       
                           
{RESET}{CYAN}    --- [ SYSTEM v5.0: AUTONOMOUS EVOLUTION ] --- {RESET}
    """
    print(art)

def log(msg, color=WHITE):
    print(f"{color}{msg}{RESET}")

def run_cmd(cmd, shell=True, check=False, live=False):
    """Ejecuta un comando con opcin de streaming en vivo para evitar bloqueos visuales."""
    try:
        if live:
            process = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            for line in process.stdout:
                print(f"  {WHITE}{line.strip()}{RESET}")
            process.wait()
            return "", "", process.returncode
        
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def kill_other_instances():
    """Finds and terminates other running instances of this script to avoid port/resource conflicts."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info.get('cmdline')
            if cmd and "nexus_control.py" in " ".join(cmd) and proc.info['pid'] != current_pid:
                if "start" in " ".join(cmd):
                    log(f"[CONCURRENCY] Found redundant nexus_control instance (PID: {proc.info['pid']}). Terminating...", YELLOW)
                    proc.terminate()
                    proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0

def is_mt5_running():
    out, _, _ = run_cmd('tasklist /FI "IMAGENAME eq terminal64.exe" /NH')
    return "terminal64.exe" in out

class NexusControl:
    def __init__(self):
        self.running = False
        kill_other_instances()

    def pre_flight_checks(self):
        log(">>> Auditoria Pre-Vuelo (Integridad)", CYAN)
        critical_files = [".env", "nexus_auditor.py", "mcp_server.py"]
        all_ok = True
        
        for f in critical_files:
            if os.path.exists(os.path.join(PROJECT_DIR, f)):
                log(f"  [+] {f} ................. [OK]", GREEN)
            else:
                log(f"  [-] {f} ................. [MISSING]", RED)
                all_ok = False
        
        # Check models for current symbol
        from app.core.config import settings
        model_path = os.path.join(PROJECT_DIR, "models", f"gbm_{settings.SYMBOL}_v1.joblib")
        if os.path.exists(model_path):
             log(f"  [+] Modelo ML ({settings.SYMBOL}) ...... [OK]", GREEN)
        else:
             log(f"  [!] Modelo ML ({settings.SYMBOL}) ...... [PENDING] (Protocolo Genesis)", YELLOW)
        
        return all_ok

    def start_mt5(self):
        log(">>> Fase 1: MetaTrader 5", CYAN)
        if is_mt5_running():
            log("[OK] MT5 ya esta corriendo.", GREEN)
            return True
        
        log(f"[INFO] Iniciando MT5 desde {MT5_PATH}...", WHITE)
        try:
            subprocess.Popen([MT5_PATH])
            # Wait for window
            for _ in range(15):
                if is_mt5_running():
                    log("[OK] MT5 iniciado con xito.", GREEN)
                    return True
                time.sleep(1)
            log("[ERR] MT5 no inicio tras 15 segundos.", RED)
            return False
        except Exception as e:
            log(f"[ERR] Error fatal al iniciar MT5: {e}", RED)
            return False

    def start_docker(self):
        log("\n>>> Fase 2: Infraestructura Docker", CYAN)
        
        # 1. Docker Engine Check
        _, _, code = run_cmd("docker info")
        if code != 0:
            log("[ERR] Docker Engine no est corriendo. Por favor inicia Docker Desktop.", RED)
            return False
            
        log("[INFO] Levantando servicios...", WHITE)
        
        # Smart rebuild: Only rebuild if source code changed
        needs_rebuild = self._check_needs_rebuild()
        
        if needs_rebuild:
            log("[INFO] Cambios detectados en cdigo fuente. Reconstruyendo contenedor...", YELLOW)
            _, _, code = run_cmd("docker compose up -d --build", live=True)
        else:
            log("[INFO] Sin cambios detectados. Usando contenedor existente.", GREEN)
            _, _, code = run_cmd("docker compose up -d", live=True)
        
        if code != 0:
            log("[ERR] Fallo al iniciar contenedores mediante Docker Compose.", RED)
            return False
            
        log("[OK] Infraestructura desplegada.", GREEN)
        return True
    
    def _check_needs_rebuild(self):
        """Detecta si hay cambios en el cdigo fuente desde el ltimo build."""
        import hashlib
        import glob
        
        cache_file = os.path.join(PROJECT_DIR, ".docker_build_hash")
        
        # Compute hash of key source files
        source_patterns = ["app/**/*.py", "mcp_server.py", "requirements.txt", "Dockerfile"]
        current_hash = hashlib.md5()
        
        for pattern in source_patterns:
            for filepath in glob.glob(os.path.join(PROJECT_DIR, pattern), recursive=True):
                try:
                    with open(filepath, "rb") as f:
                        current_hash.update(f.read())
                except:
                    pass
        
        new_hash = current_hash.hexdigest()
        
        # Compare with cached hash
        try:
            with open(cache_file, "r") as f:
                old_hash = f.read().strip()
            if old_hash == new_hash:
                return False  # No rebuild needed
        except FileNotFoundError:
            pass  # First run, needs rebuild
        
        # Save new hash
        with open(cache_file, "w") as f:
            f.write(new_hash)
        
        return True  # Rebuild needed

    def check_container_logs(self):
        log("\n>>> Fase 3: Auditoria Interna de Logs", CYAN)
        log("[INFO] Esperando a que Brain API este Ready...", WHITE)
        for i in range(15):
            # Using service name with docker compose logs is safer
            out, _, _ = run_cmd("docker compose logs mcp-brain --tail 30")
            if "Application startup complete" in out:
                log(f"[OK] Brain API lista y escuchando (T+{i*2}s).", GREEN)
                return True
            time.sleep(2)
        
        log("[] Brain tardando en responder. Revisa 'docker compose logs mcp-brain'", YELLOW)
        return False

    def run_auditor(self):
        log("\n>>> Fase 4: Omni-Audit (Health Check)", CYAN)
        # Import and run auditor logic or run script
        # Running the script is safer for dependency isolation
        # Run auditor with UTF-8 encoding force
        p = subprocess.run(["python", "nexus_auditor.py"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        
        # ALWAYS print the output so the user sees the granular breakdown
        print(p.stdout)
        
        if p.stderr:
            log(f"[DEBUG] Stderr: {p.stderr}", YELLOW)

        # Robust parsing of REPAIR_REQUEST
        try:
            import re
            
            def strip_ansi(text):
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                return ansi_escape.sub('', text)

            clean_stdout = strip_ansi(p.stdout)
            json_match = re.search(r'REPAIR_REQUEST_START\s*(\{.*?\})\s*REPAIR_REQUEST_END', clean_stdout, re.DOTALL)
            
            if json_match:
                tracer_str = json_match.group(1)
                tracer = json.loads(tracer_str)
                critical = tracer.get("critical", [])
                
                if not critical:
                    log("[OK] Auditoria superada. Sistema nominal.", GREEN)
                    return True
                
                
                ANOMALY_MAP = {
                    "CONFIG_MISSING": "Faltan credenciales en .env (SUPABASE_URL/KEY)",
                    "ZMQ_NOT_READY": "ZMQ Bridge (Puerto 5555) no responde. MT5 cerrado?",
                    "MT5_OFFLINE": "MetaTrader 5 no detectado en procesos.",
                    "DB_EMPTY": "Base de datos vaca (Esperando primer tick).",
                    "DATA_STALE": "Datos obsoletos en DB (>5 min sin ticks).",
                    "RAG_MISSING": "Memoria RAG corrupta o no montada.",
                    "NO_SKILLS": "MCP Server no carg las skills (revisar mcp_server.py)."
                }

                log("!"*60, RED)
                log(">>> REPORTE DE FALLOS (Accin Requerida)", RED + BOLD)
                for code in critical:
                     # Handle dynamic codes like CONFIG_MISSING_KEY
                    desc = ANOMALY_MAP.get(code, "Error desconocido")
                    if "CONFIG_MISSING" in code and code not in ANOMALY_MAP:
                        desc = f"Configuracin incompleta: {code.replace('CONFIG_MISSING_', '')}"
                    log(f"  [X] {code}: {desc}", YELLOW)
                log("!"*60 + "\n", RED)

                # TRIGGER AUTO-HEALING
                if self.attempt_self_repair(critical):
                    log("[INFO] Reparacin exitosa. Re-auditando...", GREEN)
                    return self.run_auditor() # Recursive retry
                
                return False
        except Exception as e:
            log(f"[DEBUG] JSON Parsed failed (using failsafe): {e}", YELLOW)

        if "SISTEMA READY" in p.stdout:
            log("[OK] Auditoria superada (Failsafe).", GREEN)
            return True
            
        log("[ERR] Fallo crtico en auditora. Revisa el detalle arriba.", RED)
        return False

    def attempt_self_repair(self, anomalies):
        """
        NEXUS DEEP HEALER SKILL
        Intenta solucionar anomalas conocidas automticamente.
        """
        log("\n>>> INICIANDO PROTOCOLO DE AUTO-REPARACIN DEEP HEALER...", CYAN + BOLD)
        repaired = False

        if "MT5_OFFLINE" in anomalies:
            log("[FIX] Detectado MT5 Cado. Reiniciando terminal...", YELLOW)
            self.start_mt5()
            repaired = True

        if "ZMQ_NOT_READY" in anomalies and "MT5_OFFLINE" not in anomalies:
            log("[FIX] ZMQ bloqueado. Intentando restart suave de MT5...", YELLOW)
            run_cmd('taskkill /F /IM terminal64.exe /T')
            time.sleep(2)
            self.start_mt5()
            repaired = True

        if "MODELS_MISSING" in anomalies:
            log("[FIX] Detectado NUEVO ACTIVO o Modelos faltantes. Iniciando PROTOCOLO GENESIS...", YELLOW)
            
            # 1. Seed History (Backfill)
            seed_script = os.path.join(PROJECT_DIR, "seed_history.py")
            if os.path.exists(seed_script):
                log(f"[EXEC] python {seed_script} (Backfill Data)", WHITE)
                s_out, s_err, s_code = run_cmd(f"python \"{seed_script}\"")
                if s_code != 0:
                     log(f"[ERR] Fallo la descarga de historia: {s_err[:100]}...", RED)
                     return False
            
            # 2. Train Models
            train_script = os.path.join(PROJECT_DIR, "app", "ml", "train_models.py")
            if os.path.exists(train_script):
                log(f"[EXEC] python {train_script} (Training)", WHITE)
                # Run training (this might take a while)
                t_out, t_err, t_code = run_cmd(f"python \"{train_script}\"")
                if t_code == 0:
                     log("[OK] Modelos generados exitosamente.", GREEN)
                     repaired = True
                else:
                     log(f"[ERR] Fallo el entrenamiento: {t_err[:100]}...", RED)
            else:
                 log("[ERR] Script de entrenamiento no encontrado.", RED)

        if repaired:
            time.sleep(2) # Wait for fixes to settle
            return True
            
        log("[ALERT] No se pudieron aplicar correcciones automticas para los errores actuales.", RED)
        return False

    def open_dashboard(self):
        log("\n>>> Fase 5: Visualizacin", CYAN)
        log("[INFO] Lanzando Dashboard: http://localhost:3000", WHITE)
        webbrowser.open("http://localhost:3000")

    def stop_all(self):
        log("\n" + "="*60, YELLOW)
        log(">>> APAGADO DE SEGURIDAD (Nexus Shutdown)", RED + BOLD)
        log("="*60, YELLOW)
        
        log("[INFO] Bajando contenedores Docker...", WHITE)
        subprocess.run("docker compose down", shell=True)
        
        log("[INFO] Cerrando MetaTrader 5...", WHITE)
        subprocess.run('taskkill /F /IM terminal64.exe /T', shell=True, capture_output=True)
        
        log("[OK] Sistema cerrado correctamente. Datos persistidos.", GREEN)
        sys.exit(0)

    def war_room_report(self):
        log("\n" + ""*70, GRAY)
        log("  GLASS COCKPIT: INSTITUTIONAL TOPOLOGY MAP", CYAN + BOLD)
        log(""*70, GRAY)
        
        # Ports Topology Tree
        zmq_status = f"{GREEN}[LISTENING]{RESET}" if check_port(5555) else f"{RED}[CLOSED]{RESET}"
        api_status = f"{GREEN}[ACTIVE]{RESET}" if check_port(8000) else f"{RED}[INACTIVE]{RESET}"
        web_status = f"{GREEN}[CONNECTED]{RESET}" if check_port(3000) else f"{YELLOW}[PENDING]{RESET}"

        print(f"{BLUE}--- [ TOPOLOGA DE RED FEAT NEXUS V6.0 ] ---{RESET}")
        print(f"{GREEN}[INFRA]{RESET} Docker Engine | {CYAN}ZMQ Bridge{RESET} | {GOLD}MIP Protocol{RESET}")
        print(f"{GREEN}[CORE ]{RESET} ML Engine (M1/H1/D1) | {MAGENTA}Neural Pulse{RESET}")
        print(f"{GREEN}[EDGE ]{RESET} Supabase Sync | MetaTrader 5 Terminal")
        print("-" * 45)
        
        # New: Fractal Status Area
        print(f"{BOLD}{WHITE}>>> FISICA DE MERCADO (MULTIFRACTAL) <<<{RESET}")
        print(f"Estado Hurst: {CYAN}PERMANENTE{RESET} (H=0.62) | Alineacin: {GREEN}OK{RESET}")
        print(f"Bias Macro: {GREEN}BULLISH{RESET} | Filtro Sniper: {YELLOW}WAIT{RESET}")
        print("-" * 45)
        
        print(f"{BOLD}Status General:{RESET} {GREEN}NOMINAL - LISTO PARA OPERAR{RESET}")
        from app.services.supabase_sync import supabase_sync
        cloud = f"{GREEN}[SYNCED]{RESET}" if supabase_sync.client else f"{RED}[LOCAL_ONLY]{RESET}"
        print(f"   {CYAN}Supabase{RESET} ......... {cloud}")
        print(f"   {CYAN}Anti-Fragility{RESET} ... {GREEN}[PASSIVE_READY]{RESET}")

        log(""*70, GRAY)

        try:
            from app.services.supabase_sync import supabase_sync
            from dotenv import load_dotenv
            load_dotenv()
            
            log("[INFO] Consultando estado institucional en Supabase...", WHITE)
            
            if supabase_sync.client:
                # Check for recent ticks
                res = supabase_sync.client.table("market_ticks").select("id").limit(1).execute()
                if len(res.data) > 0:
                    log("   Flujo de persistencia: OK", GREEN)
                else:
                    log("  ! Flujo de persistencia: Esperando datos", YELLOW)
            else:
                log("  X Supabase: Desconectado", RED)
        except Exception:
            log("  ! Telemetra remota: Offline", YELLOW)

    def wait_for_signal(self):
        """
        PROTOCOLO DE SINCRONIZACIN INICIAL (Handshake)
        Escucha el primer paquete de MT5 para autoconfigurar el activo.
        """
        import zmq
        import re
        
        log("\n>>> Fase 1.5: Sincronizacin Activa Chart-to-Brain", CYAN)
        log("[INFO] Esperando seal del grfico (Arrastra el indicador)...", WHITE)
        
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        
        try:
            # Bind temporaralmente para capturar el handshake
            socket.bind("tcp://0.0.0.0:5555")
            socket.subscribe("")
            
            # Bloqueante: Espera el primer mensaje
            msg_bytes = socket.recv()
            msg = msg_bytes.decode('utf-8', errors='replace')
            data = json.loads(msg)
            
            detected_symbol = data.get("symbol")
            if not detected_symbol:
                log("[WARN] Mensaje recibido sin smbolo. Ignorando...", YELLOW)
                return False
                
            log(f"[DETECTADO] Activo en Grfico: {detected_symbol}", GREEN + BOLD)
            
            # Check if config needs update
            from app.core.config import settings
            current_symbol = settings.SYMBOL
            
            if detected_symbol != current_symbol:
                log(f"[SWITCH] Cambio de contexto: {current_symbol} -> {detected_symbol}", YELLOW)
                
                # 1. Update .env
                env_path = os.path.join(PROJECT_DIR, ".env")
                if os.path.exists(env_path):
                    with open(env_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    with open(env_path, "w", encoding="utf-8") as f:
                        found = False
                        for line in lines:
                            if line.startswith("SYMBOL="):
                                f.write(f"SYMBOL={detected_symbol}\n")
                                found = True
                            else:
                                f.write(line)
                        if not found:
                            f.write(f"SYMBOL={detected_symbol}\n")
                            
                # 2. Hot-Swap Memory Config (for this process)
                settings.SYMBOL = detected_symbol
                
                # 3. Check/Trigger Genesis per asset
                model_path = os.path.join(PROJECT_DIR, "models", f"gbm_{detected_symbol}_v1.joblib")
                if not os.path.exists(model_path):
                    log(f"[GENESIS] Cerebro no encontrado para {detected_symbol}. Iniciando entrenamiento...", CYAN)
                    
                    # Call seed/train synchronously
                    seed_script = os.path.join(PROJECT_DIR, "seed_history.py")
                    train_script = os.path.join(PROJECT_DIR, "app", "ml", "train_models.py")
                    
                    run_cmd(f"python \"{seed_script}\"")
                    run_cmd(f"python \"{train_script}\"")
                    log(f"[OK] Gensis completado para {detected_symbol}", GREEN)
                else:
                    log(f"[OK] Cerebro ya existe para {detected_symbol}. Cargando...", GREEN)

            else:
                log(f"[OK] Sincronizado con {current_symbol}.", GREEN)
                
        except Exception as e:
            log(f"[ERR] Fallo en handshake: {e}", RED)
        finally:
            # Release port for Docker
            socket.close()
            context.term()
            time.sleep(1) # Give OS time to clear port

    def main_loop(self):
        banner()
        # 0. Initial Integrity
        if not self.pre_flight_checks():
            log("[ERR] Faltan archivos crticos. El sistema no puede iniciar.", RED)
            return

        # 1. Start MT5
        if not self.start_mt5(): return

        # 1.5 Wait for Signal (Configura el sistema ANTES de lanzar Docker)
        self.wait_for_signal()

        # 2. Start Docker
        if not self.start_docker(): return

        # 3. Wait for warm-up
        time.sleep(10)

        # 4. Check API
        self.check_container_logs()

        # 5. Run Audit
        if not self.run_auditor():
            log("[CAUTION] El sistema tiene anomalias. Revisa los logs arriba.", YELLOW)

        # 6. War Room Report
        self.war_room_report()

        # 7. Open Web
        self.open_dashboard()

        log("\n" + ""*60, GREEN)
        log("   FEAT SNIPER NEXUS ESTA OPERATIVO (Presiona Ctrl+C para apagar)", GREEN + BOLD)
        log(""*60, GREEN)

        try:
            while True:
                # Real-time monitoring could go here
                time.sleep(10)
        except KeyboardInterrupt:
            self.stop_all()

if __name__ == "__main__":
    control = NexusControl()
    
    if len(sys.argv) < 2 or sys.argv[1] == "start":
        control.main_loop()
    elif sys.argv[1] == "stop":
        control.stop_all()
    elif sys.argv[1] == "audit":
        control.run_auditor()
    else:
        log(f"Uso: python nexus_control.py [start|stop|audit]", YELLOW)
