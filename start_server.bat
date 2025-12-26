@echo off
setlocal enabledelayedexpansion

:: =============================================================================
:: MT5 NEURAL BRIDGE - SENIOR ENTRYPOINT v2.1
:: =============================================================================
title MT5 Neural Bridge - Enterprise Server
mode con: cols=100 lines=30
color 0A

echo ===========================================================================
echo                 MT5 NEURAL BRIDGE - FEAT SNIPER 2.0 
echo ===========================================================================
echo.

:: 1. Verificacion de Directorio
cd /d "%~dp0"
echo [STEP 1] Validando entorno local...

:: 2. Verificacion de .env
if not exist ".env" (
    echo [FATAL] Archivo .env no encontrado.
    echo [INFO] Copia .env.example a .env y configura tus credenciales de MT5.
    goto :error_exit
)
echo      - Archivo .env: OK

:: 3. Gestion de Port 8000
echo [STEP 2] Verificando disponibilidad de red - Port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo      - Port 8000 ocupado por PID %%a. Matando proceso previo...
    taskkill /F /PID %%a >nul 2>&1
)
:: Limpieza extra de procesos
taskkill /F /FI "WINDOWTITLE eq MT5 Neural Bridge*" /IM python.exe >nul 2>&1

:: 4. Virtual Environment Detection
if exist ".venv\Scripts\activate.bat" (
    echo [STEP 3] Activando entorno virtual .venv...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [STEP 3] Activando entorno virtual venv...
    call venv\Scripts\activate.bat
) else (
    echo [STEP 3] Aviso: No se detecto .venv local. Usando Python Global.
)

:: 5. Pre-flight Python Validation
echo [STEP 4] Validando dependencias criticas...
python -c "import fastapi, uvicorn, MetaTrader5" >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Faltan dependencias criticas - fastapi, uvicorn o MetaTrader5.
    echo [HINT] Ejecuta: pip install -r requirements.txt
    goto :error_exit
)
echo      - Modulos core: OK

:: 6. Lanzamiento
echo.
echo [SUCCESS] Todo listo. Iniciando Gateway...
echo ---------------------------------------------------------------------------
echo.

python -u run.py

if %errorlevel% neq 0 goto :error_exit
exit /b

:error_exit
echo.
echo ===========================================================================
echo [STOP] El servidor no pudo iniciarse o se detuvo inesperadamente.
echo ===========================================================================
echo.
echo Sugerencias de Troubleshooting:
echo 1. Verifica que MetaTrader 5 este abierto y conectado a tu cuenta.
echo 2. Revisa que el puerto 8000 no este bloqueado por un Firewall.
echo 3. Asegurate de tener el Terminal64.exe en el PATH.
echo.
echo Presiona cualquier tecla para salir...
pause >nul
exit /b
