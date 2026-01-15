import sys
import os
import json
import subprocess
import time
import warnings

# Filter noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from pydantic.warnings import PydanticDeprecatedSince212
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince212)
except ImportError:
    pass

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

def check_mql5_compilation():
    log("[AUDIT] >> Verificando Sincronizacion MQL5/EX5...", YELLOW)
    search_dir = "FEAT_Sniper_Master_Core"
    if not os.path.exists(search_dir):
        # Fallback search in root
        search_dir = "."
        
    outdated_count = 0
    
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith(".mq5"):
                mq5_path = os.path.join(root, file)
                ex5_path = mq5_path.replace(".mq5", ".ex5")
                
                if not os.path.exists(ex5_path):
                    log(f"[WARN] Sin compilar: {file} (No existe .ex5)", RED)
                    outdated_count += 1
                else:
                    mq5_mtime = os.path.getmtime(mq5_path)
                    ex5_mtime = os.path.getmtime(ex5_path)
                    
                    if mq5_mtime > ex5_mtime:
                        log(f"[WARN] Desactualizado: {file} (Codigo mas nuevo que binario)", RED)
                        outdated_count += 1
                        
    if outdated_count == 0:
        log("[OK] Indicadores MQL5 Sincronizados.", GREEN)
    else:
        log(f"[ALERTA] Se encontraron {outdated_count} indicadores desactualizados. Recompilar en MetaEditor.", RED)

def check_model_freshness():
    log("[AUDIT] >> Verificando Frescura del Modelo Neuronal...", YELLOW)
    model_path = "models/feat_hybrid_v2.pth"
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        dt = datetime.fromtimestamp(mtime)
        age_hours = (datetime.now() - dt).total_seconds() / 3600
        
        color = GREEN if age_hours < 24 else YELLOW
        log(f"   - Modelo Activo: {model_path}", color)
        log(f"   - Ultima Modificacion: {dt} ({age_hours:.1f} horas atras)", color)
    else:
        log("[WARN] No se encontro modelo neuronal (.pth). Se usara inicializacion aleatoria.", RED)

if __name__ == "__main__":
    from datetime import datetime
    log("=== NEXUS AUDITOR v2.1 (Deep Scan) ===", GREEN)
    
    # 1. Run The Cartographer
    run_cartographer()
    
    # 2. Analyze Results
    analyze_map()
    
    # 3. Version Checks (New)
    check_mql5_compilation()
    check_model_freshness()

    
    # 3. Final Verification
    log("\n[OK] Auditoría Finalizada.", GREEN)
    sys.exit(0)
