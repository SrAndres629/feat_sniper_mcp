"""
Gemini CLI Bulk Code Improver
=============================
Automatiza la mejora de todos los archivos Python usando Gemini CLI.
"""
import os
import subprocess
import sys
from pathlib import Path
import time

# Configuración
PROJECT_ROOT = Path(__file__).parent.parent  # Subir de tools/ a feat_sniper_mcp/
EXCLUDED_DIRS = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', 'logs', 'tests'}
EXCLUDED_FILES = {'__init__.py', 'conftest.py', 'setup.py'}

def get_python_files(root: Path):
    """Obtiene todos los archivos Python del proyecto."""
    files = []
    for path in root.rglob('*.py'):
        # Excluir directorios
        if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
            continue
        # Excluir archivos
        if path.name in EXCLUDED_FILES:
            continue
        files.append(path)
    return sorted(files)

def run_gemini_audit(file_path: Path) -> dict:
    """Ejecuta Gemini CLI para auditar un archivo."""
    prompt = f"Audita el archivo Python '{file_path.name}'. Busca: 1) Errores, 2) Imports faltantes, 3) Mejoras de seguridad. Responde en JSON con keys: issues (list), improvements (list), fixed_code (string o null si no hay cambios)."
    
    try:
        result = subprocess.run(
            ['gemini', '-p', prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60,
            cwd=str(PROJECT_ROOT)
        )
        
        if result.returncode == 0:
            return {
                'file': str(file_path),
                'status': 'success',
                'output': result.stdout[:500]  # Limitar output
            }
        else:
            return {
                'file': str(file_path),
                'status': 'error',
                'error': result.stderr[:200]
            }
    except subprocess.TimeoutExpired:
        return {'file': str(file_path), 'status': 'timeout'}
    except Exception as e:
        return {'file': str(file_path), 'status': 'exception', 'error': str(e)}

def main():
    print("=" * 60)
    print("  GEMINI CLI BULK CODE IMPROVER")
    print("=" * 60)
    
    files = get_python_files(PROJECT_ROOT)
    print(f"\n📁 Encontrados {len(files)} archivos Python para auditar.\n")
    
    if '--dry-run' in sys.argv:
        print("DRY RUN - Solo mostrando archivos:")
        for f in files:
            print(f"  - {f.relative_to(PROJECT_ROOT)}")
        return
    
    results = []
    for i, file_path in enumerate(files, 1):
        rel_path = file_path.relative_to(PROJECT_ROOT)
        print(f"[{i}/{len(files)}] 🔍 Auditando: {rel_path}")
        
        result = run_gemini_audit(file_path)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"        ✅ Completado")
        else:
            print(f"        ⚠️ {result['status']}: {result.get('error', '')[:50]}")
        
        # Rate limiting
        time.sleep(1)
    
    # Resumen
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    success = sum(1 for r in results if r['status'] == 'success')
    print(f"✅ Exitosos: {success}/{len(results)}")
    print(f"⚠️ Errores: {len(results) - success}/{len(results)}")

if __name__ == '__main__':
    main()
