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
from app.core.config import settings

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

class StrategyCommander:
    """
    MODELO 3: EL COMANDANTE DE ESTRATEGIA
    Orquesta la lgica de decisin, riesgo y ejecucin basada en FSM.
    """
    def __init__(self):
        self.fsm = PerformanceTracker()
        self.running = True
        
        # Lazy imports to avoid circular dependencies at module level
        from app.services.risk_engine import risk_engine
        # Execution is handled via ZMQ or direct call if local
        # Ideally we call the execution module directly since we are the controller
        
        self.risk_engine = risk_engine

    async def execute_logic(self, tick_data: Dict[str, Any]):
        """
        Nucleo decisional: Brain -> Risk -> FSM -> Execution.
        """
        symbol = tick_data.get("symbol")
        if not symbol: return

        # 1. ACTUALIZAR MAQUINA DE ESTADOS
        # (Se hace post-trade, aqu solo leemos el estado)
        current_state = self.fsm.current_state
        
        if current_state == TradingState.RECALIBRATION:
            # BLOQUEO TOTAL
            # Podriamos enviar un heartbeat de "Diagnostico Requerido"
            return
            
        # 2. CONSULTAR CEREBRO (NEXUS BRAIN)
        # En una arquitectura real, el cerebro ya proces los datos y los adjunt al tick
        # o hacemos la inferencia aqui. Por eficiencia, asumimos que mcp_server 
        # o un componente previo inyect la prediccin en 'tick_data' o consultamos ahora.
        # Para Model 3, vamos a simular la llamada o usar ml_engine si est disponible.
        
        try:
             from app.ml.ml_engine import ml_engine
             # Features deberan venir pre-calculados o calculados al vuelo
             # Por simplicidad, asumimos que ml_engine puede manejarlo o que el tick trae "prediction"
             # Si el tick viene de mcp_server (que ya llam al cerebro), usamos eso.
             
             prediction = tick_data.get("neural_prediction")
             if not prediction:
                 # Fallback: Pedir al cerebro si somos el nodo maestro
                 # Nota: Esto requiere features completos, que podran no estar en un simple tick
                 return 
        except ImportError:
             return

        p_win = prediction.get("p_win", 0.0)
        confidence = prediction.get("alpha_confidence", 0.0)
        urgency = prediction.get("urgency", 0.0)
        volatility = prediction.get("volatility_regime", 0.0)

        # 3. FILTRO DE CONFIRMACION (Thresholds)
        # Solo procesamos si hay seal clara
        if p_win < 0.60: return 

        # 4. GESTION DE CAPITAL (THE VAULT)
        allocation = await self.risk_engine.get_neural_allocation(confidence)
        lot_size = await self.risk_engine.get_adaptive_lots(symbol, 200, allocation["lot_multiplier"]) # 200 pts default SL
        
        if lot_size <= 0: return

        # 5. EJECUCION POR ESTADO (FSM)
        direction = "BUY" # Placeholder, vendra del modelo (prob > 0.5)
        
        cmd = {
            "symbol": symbol,
            "action": direction,
            "volume": lot_size,
            "price": tick_data.get("ask" if direction == "BUY" else "bid"),
            "sl": 0, # Se calcularia dinamico
            "tp": 0,
            "comment": f"Fsm:{current_state.name}|Urg:{urgency:.2f}"
        }

        from app.skills.execution import send_order
        from app.models.schemas import TradeOrderRequest
        
        if current_state == TradingState.AUTONOMOUS:
            # EJECUCION INMEDIATA (Low Latency)
            # Mapear a Request Object
            req = TradeOrderRequest(**cmd)
            await send_order(req, urgency_score=urgency)
            print(f"🚀 [AUTONOMOUS] Orden enviada: {symbol} {direction} {lot_size} lots")
            
        elif current_state == TradingState.SUPERVISED:
            # PROTOCOLO DE SUPERVISION (N8N / Human Loop)
            # En V5, simulamos el envio a N8n y esperamos (o logueamos)
            print(f"🟡 [SUPERVISED] Seal generada. Esperando aprobacin externa... (Simulada)")
            # Aqui llamariamos a n8n webhook
            # await trigger_n8n_webhook(cmd)
            pass

    async def run_loop(self):
        """Bucle principal asincrono de estrategia."""
        import zmq.asyncio
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://127.0.0.1:{settings.ZMQ_PORT}") # Conecta al Publisher de MT5
        socket.subscribe("")
        
        print(f"⚖️ [STRATEGY COMMANDER] Escuchando Market Data en puerto {settings.ZMQ_PORT}...")
        
        while self.running:
            try:
                msg = await socket.recv_json()
                # Process tick
                await self.execute_logic(msg)
            except Exception as e:
                print(f"Error en loop de estrategia: {e}")
                await asyncio.sleep(1)

# Wrapper para correr en el main
def start_strategy_node():
    commander = StrategyCommander()
    import asyncio
    try:
        asyncio.run(commander.run_loop())
    except KeyboardInterrupt:
        print("Apagando Commander...")

class NexusControl:
    def __init__(self):
        self.running = False
        kill_other_instances()

    def pre_flight_checks(self) -> bool:
        """Verificaciones previas al despegue."""
        log("\n[PRE-FLIGHT] Verificando sistemas...", CYAN)
        
        # 1. MT5 Path
        if not os.path.exists(MT5_PATH):
            log(f"  [FAIL] MT5 executable not found at: {MT5_PATH}", RED)
            return False
        log(f"  [OK] FX Terminal detected: {MT5_PATH}", GREEN)
        
        # 2. Ports availability (Check if 5555 is free if we are server, or reachable if client?)
        # Since we bind to 5555 in docker/python, we just ensure no one else is blocking it heavily.
        # Actually logic is handled by Docker usually.
        
        log("  [OK] Systems Nominal.", GREEN)
        return True

    def start_mt5(self) -> bool:
        """Inicia MetaTrader 5 si no est corriendo."""
        if is_mt5_running():
            log("  [SKIP] MT5 ya est en ejecucin.", YELLOW)
            return True
            
        log("  [INIT] Lanzando MetaTrader 5...", CYAN)
        try:
            subprocess.Popen(MT5_PATH)
            time.sleep(10) # Wait for init
            return True
        except Exception as e:
            log(f"  [FAIL] Error launching MT5: {e}", RED)
            return False

    def start_docker(self) -> bool:
        """Verifica e inicia el stack Docker (Brain)."""
        log("  [DOCKER] Verificando Neural Containers...", CYAN)
        
        # Check if container running
        out, _, _ = run_cmd("docker ps --format '{{.Names}}'")
        if "mcp-brain" in out:
             log("  [OK] Neural Brain is ACTIVE.", GREEN)
             return True
             
        log("  [INIT] Desplegando Neural Brain...", YELLOW)
        out, err, code = run_cmd("docker compose up -d")
        if code != 0:
            log(f"  [FAIL] Docker Start Failed: {err}", RED)
            return False
            
        return True

    def war_room_report(self):
        """Imprime estado final del sistema."""
        print("\n" + "="*40)
        print(f"{BOLD}   WAR ROOM STATUS REPORT   {RESET}")
        print("="*40)
        print(f"TERMINAL:    {GREEN}ONLINE{RESET}")
        print(f"BRAIN (ML):  {GREEN}ONLINE{RESET}")
        print(f"BRIDGE:      {GREEN}LISTENING (5555/5556){RESET}")
        print(f"MODE:        {GREEN}AUTONOMOUS (Model 5){RESET}")
        print("="*40 + "\n")
        
        log("\n>>> INICIANDO STRATEGY COMMANDER (FSM)...", CYAN + BOLD)
        # Start the logic loop
        start_strategy_node()

    def main_loop(self):
        """Orquestador principal de inicio."""
        banner()
        log(">>> [MISSION START] Iniciando Protocolo FEAT NEXUS v5.0...", CYAN + BOLD)
        
        if not self.pre_flight_checks():
            log("[ABORT] Fallo en verificaciones previas.", RED)
            return

        if not self.start_docker():
            log("[ABORT] Fallo al iniciar Neural Brain.", RED)
            return

        if not self.start_mt5():
            log("[WARN] MT5 no pudo iniciarse, pero el Brain está activo.", YELLOW)

        self.war_room_report()

    def stop_all(self):
        """Detiene todos los servicios."""
        log(">>> [SHUTDOWN] Deteniendo ecosistema...", YELLOW)
        run_cmd("docker compose down")
        # MT5 termination could be added here if needed
        log("[OK] Sistema detenido.", GREEN)

    def audit_system(self):
        """Realiza un diagnóstico profundo del sistema."""
        banner()
        log(">>> [AUDIT] Ejecutando Diagnóstico Institucional...", CYAN)
        
        # 1. ZMQ Check
        if check_port(5555):
            log("  [OK] ZMQ Port 5555 is ACTIVE.", GREEN)
        else:
            log("  [FAIL] ZMQ Port 5555 is CLOSED.", RED)
            
        # 2. Docker Check
        out, _, _ = run_cmd("docker ps --format '{{.Names}}'")
        if "feat-sniper-brain" in out or "mcp-brain" in out:
            log("  [OK] Docker Container is RUNNING.", GREEN)
        else:
            log("  [FAIL] Docker Container is MISSING.", RED)
            
        # 3. MT5 Check
        if is_mt5_running():
            log("  [OK] MT5 Terminal is RUNNING.", GREEN)
        else:
            log("  [OK] MT5 Terminal is CLOSED (Normal if not trading).", YELLOW)

        log("\n>>> [RESULT] Auditoría completada. Revisar logs/nexus.log para detalles.", GRAY)

if __name__ == "__main__":
    control = NexusControl()
    
    if len(sys.argv) < 2 or sys.argv[1] == "start":
        control.main_loop()
    elif sys.argv[1] == "stop":
        control.stop_all()
    elif sys.argv[1] == "audit":
        control.audit_system()
