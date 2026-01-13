import sys
import os
import json
import subprocess
import time

# Colors
GREEN = "\033[38;5;82m"
RED = "\033[38;5;196m"
YELLOW = "\033[38;5;214m"
RESET = "\033[0m"

def log(msg, color=RESET):
    print(f"{color}{msg}{RESET}")

def run_cartographer():
    log("\n[AUDIT] >> Ejecutando Nexus Cartographer...")
    try:
        subprocess.run([sys.executable, "tools/map_project.py"], check=True)
    except subprocess.CalledProcessError:
        log("[FAIL] El Cartografo fallo critico.", RED)
        sys.exit(1)

def analyze_map():
    log("[AUDIT] >> Analizando Integridad Estructural...")
    if not os.path.exists("architecture_map.json"):
        log("[FAIL] No se encontro el mapa de arquitectura.", RED)
        sys.exit(1)

    with open("architecture_map.json", "r") as f:
        data = json.load(f)

    orphans = data.get("orphans", [])
    
    # 1. Check for Orphans in Critical Zones
    critical_orphans = []
    for orphan in orphans:
        if "core" in orphan or "skills" in orphan:
            critical_orphans.append(orphan)

    if critical_orphans:
        log(f"[ALERTA] !! CODIGO MUERTO DETECTADO ({len(critical_orphans)} archivos):", RED)
        for o in critical_orphans:
            print(f"   - {o}")
        
        # Strict Mode: Exit if core/skills have orphans?
        # For now, just a loud warning. user asked to throw ERROR/WARNING.
        # log("[FAIL] La integridad arquitectonica esta comprometida.", RED)
        # sys.exit(1) 
        log("[WARN] Se detectaron modulos huerfanos. Verificar antes de produccion.", YELLOW)
    else:
        log("[OK] Integridad Estructural Verificada (0 Orphans).", GREEN)

def check_ports():
    log("[AUDIT] >> Verificando Puertos (Legacy Check)...")
    # This logic keeps existing checks if needed
    pass

if __name__ == "__main__":
    log("=== NEXUS AUDITOR v2.0 ===", GREEN)
    
    # 1. Run The Cartographer
    run_cartographer()
    
    # 2. Analyze Results
    analyze_map()
    
    # 3. Final Verification
    log("\n[OK] Auditoría Finalizada.", GREEN)
    sys.exit(0)
