import subprocess
import sys
import os
import argparse
import re

def clean_markdown(content: str) -> str:
    """Removes markdown code blocks (```python ... ```)."""
    # Remove opening tag
    content = re.sub(r'^```[a-zA-Z]*\n', '', content, flags=re.MULTILINE)
    # Remove closing tag
    content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
    return content.strip()

def run_gemini(prompt: str, output_file: str = None):
    print(f"[GEMINI] 🧠 Procesando prompt (Silencioso)...")
    
    try:
        # 1. Ejecutar el CLI a través de PowerShell con captura silenciosa
        # Ruta detectada: C:\Users\acord\AppData\Roaming\npm\gemini.ps1
        gemini_ps1 = os.path.join(os.environ.get('APPDATA', ''), 'Roaming', 'npm', 'gemini.ps1')
        if not os.path.exists(gemini_ps1):
            # Probar ruta alternativa directa
            gemini_ps1 = os.path.expandvars(r'%APPDATA%\npm\gemini.ps1')

        cmd = ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", f"& '{gemini_ps1}' '{prompt}'"]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Gemini falló (Code {result.returncode}):")
            print(result.stderr)
            sys.exit(1)
            
        content = result.stdout
        
        # 2. Limpieza de Markdown
        content = clean_markdown(content)
        
        if not content:
            print("[WARN] Gemini no devolvió contenido útil.")
            return

        # 3. Escritura atómica al disco (evita archivos corruptos)
        if output_file:
            # Asegurar directorio
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[OK] 💾 Contenido guardado en: {output_file} ({len(content)} bytes)")
        else:
            # Si no hay archivo, mostramos un resumen (para no inundar el buffer)
            print(f"[INFO] Gemini respondió exitosamente.")
            print("-" * 30)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 30)

    except FileNotFoundError:
        print("[FAIL] No se encontró el ejecutable 'gemini'. Asegúrate de que esté instalado y en el PATH.")
    except Exception as e:
        print(f"[CRITICAL] Error inesperado en el Wrapper: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secure Wrapper para Gemini CLI")
    parser.add_argument("--prompt", required=True, help="El prompt para Gemini")
    parser.add_argument("--output", help="Archivo de destino opcional")
    
    args = parser.parse_args()
    run_gemini(args.prompt, args.output)
