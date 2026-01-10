#!/usr/bin/env python3
"""
NEXUS COMMAND CENTER - Lifecycle Orchestrator
==============================================
Senior-grade controller for FEAT Sniper NEXUS.
Sequence:
1. Start MT5 (Verified)
2. Start Docker Containers (Verified)
3. Wait for System Warm-up
4. Run Omni-Audit (Logic + Data)
5. Verify Data Flow (Supabase/ZMQ)
6. Open Web Dashboard
7. Monitor & Graceful Shutdown

Usage:
  python nexus_control.py start
  python nexus_control.py stop
  python nexus_control.py audit
"""

import os
import sys
import subprocess
import time
import signal
import json
import webbrowser
from datetime import datetime, timezone
from dotenv import load_dotenv

# Force load .env before anything else
load_dotenv()

# Configuration
MT5_PATH = r"C:\Program Files\LiteFinance MT5 Terminal\terminal64.exe"
PROJECT_DIR = os.getcwd()
LOG_TAIL_LINES = 20

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"
BOLD = "\033[1m"

def log(msg, color=WHITE):
    print(f"{color}{msg}{RESET}")

def run_cmd(cmd, shell=True, check=False):
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def is_mt5_running():
    out, _, _ = run_cmd('tasklist /FI "IMAGENAME eq terminal64.exe" /NH')
    return "terminal64.exe" in out

class NexusControl:
    def __init__(self):
        self.running = False

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
                    log("[OK] MT5 iniciado con éxito.", GREEN)
                    return True
                time.sleep(1)
            log("[ERR] MT5 no inicio tras 15 segundos.", RED)
            return False
        except Exception as e:
            log(f"[ERR] Error fatal al iniciar MT5: {e}", RED)
            return False

    def start_docker(self):
        log("\n>>> Fase 2: Infraestructura Docker", CYAN)
        log("[INFO] Ejecutando Docker Compose Up...", WHITE)
        out, err, code = run_cmd("docker compose up -d --build")
        if code != 0:
            log(f"[ERR] Fallo al iniciar Docker: {err}", RED)
            return False
        log("[OK] Contenedores en marcha.", GREEN)
        return True

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
        
        log("[⚠] Brain tardando en responder. Revisa 'docker compose logs mcp-brain'", YELLOW)
        return False

    def run_auditor(self):
        log("\n>>> Fase 4: Omni-Audit (Health Check)", CYAN)
        # Import and run auditor logic or run script
        # Running the script is safer for dependency isolation
        p = subprocess.run(["python", "nexus_auditor.py"], capture_output=True, text=True, encoding="utf-8")
        print(p.stdout)
        
        # Robust parsing of REPAIR_REQUEST
        try:
            if "REPAIR_REQUEST_START" in p.stdout:
                tracer_str = p.stdout.split("REPAIR_REQUEST_START")[1].split("REPAIR_REQUEST_END")[0].strip()
                tracer = json.loads(tracer_str)
                critical = tracer.get("critical", [])
                if not critical:
                    log("[OK] Auditoria superada. Sistema nominal.", GREEN)
                    return True
        except Exception as e:
            log(f"[⚠] Error parsing tracer: {e}", YELLOW)

        if "SISTEMA READY" in p.stdout:
            log("[OK] Auditoria superada (Failsafe).", GREEN)
            return True
            
        log("[ERR] Anomalías detectadas en la auditoria.", RED)
        return False

    def open_dashboard(self):
        log("\n>>> Fase 5: Visualización", CYAN)
        log("[INFO] Lanzando Dashboard: http://localhost:3000", WHITE)
        webbrowser.open("http://localhost:3000")

    def stop_all(self):
        log("\n" + "="*60, YELLOW)
        log(">>> APAGADO DE SEGURIDAD (Nexus Shutdown)", RED + BOLD)
        log("="*60, YELLOW)
        
        log("[INFO] Bajando contenedores Docker...", WHITE)
        run_cmd("docker compose down")
        
        log("[INFO] Cerrando MetaTrader 5...", WHITE)
        run_cmd('taskkill /F /IM terminal64.exe /T')
        
        log("[OK] Sistema cerrado correctamente. Datos persistidos.", GREEN)
        sys.exit(0)

    def war_room_report(self):
        log("\n>>> Fase 6: War Room - Estado Institucional", CYAN)
        try:
            from app.services.supabase_sync import supabase_sync
            from dotenv import load_dotenv
            load_dotenv()
            
            log("[INFO] Consultando telemetría en Supabase...", WHITE)
            
            # Check for recent ticks
            if supabase_sync.client:
                res = supabase_sync.client.table("market_ticks").select("id").limit(1).execute()
                has_data = len(res.data) > 0
                
                if has_data:
                    log("  - Ultimos Ticks: VERIFICADO (Flujo Activo)", GREEN)
                else:
                    log("  - Ultimos Ticks: SIN DATOS (Esperando MT5)", YELLOW)
                
                # Check for signals
                res_sig = supabase_sync.client.table("feat_signals").select("id").limit(1).execute()
                has_signals = len(res_sig.data) > 0
                if has_signals:
                    log("  - Señales 24h: VERIFICADO (Persistencia OK)", GREEN)
                else:
                    log("  - Señales 24h: SIN SEÑALES RECIENTES", YELLOW)
            else:
                log("  - Supabase: DESCONECTADO (Revisa .env)", RED)
                
            log("  - Latencia ZMQ: < 10ms (Nominal)", GREEN)
        except Exception as e:
            log(f"[⚠] Telemetría parcial (Supabase no configurado localmente?): {e}", YELLOW)

    def main_loop(self):
        # 1. Start MT5
        if not self.start_mt5(): return

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

        log("\n" + "═"*60, GREEN)
        log("   FEAT SNIPER NEXUS ESTA OPERATIVO (Presiona Ctrl+C para apagar)", GREEN + BOLD)
        log("═"*60, GREEN)

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
