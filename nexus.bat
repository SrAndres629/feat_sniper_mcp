@echo off
setlocal
chcp 65001 >nul

REM --- ZONA DE SILENCIO (Suprimir warnings de pyiceberg/pydantic) ---
set PYTHONWARNINGS=ignore
set PYTHONDONTWRITEBYTECODE=1

cls
echo ==================================================
echo   NEXUS PROTOCOL - SYSTEM BOOT SEQUENCE
echo ==================================================

REM --- FASE 0: ZOMBIE KILLER (Limpiar instancias anteriores) ---
echo [0/5] Limpiando instancias anteriores...
REM Matar procesos en puertos ZMQ
powershell -Command "Get-NetTCPConnection -LocalPort 5555 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
powershell -Command "Get-NetTCPConnection -LocalPort 5556 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
REM Matar TODAS las instancias de Python (limpieza agresiva)
taskkill /F /IM python.exe /T >nul 2>&1
echo [INFO] Esperando liberacion de puertos (2s)...
timeout /t 2 /nobreak >nul
echo [OK] Sistema limpio. Iniciando boot fresco.

REM --- FASE 1: ENTORNO ---
if exist ".venv\Scripts\activate.bat" goto :ACTIVATE_VENV
if exist "venv\Scripts\activate.bat" goto :ACTIVATE_V
echo [WARN] No se detecta entorno virtual local.
goto :RUN_AUDIT

:ACTIVATE_VENV
echo [1/4] Activando entorno neuronal (.venv)...
call .venv\Scripts\activate.bat
goto :RUN_AUDIT

:ACTIVATE_V
echo [1/4] Activando entorno neuronal (venv)...
call venv\Scripts\activate.bat

:RUN_AUDIT
REM --- FASE 1.5: LOCAL BRAIN CHECK ---
echo [1.5/5] Verificando Cerebro Local (PyTorch)...
python -c "import torch; print('Cerebro Local OK')" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] PyTorch no detectado. El Cerebro funciona en modo degradado.
)

REM --- FASE 2: AUDITORIA PROFUNDA (CARTOGRAFO) ---
echo [2/5] Generando Mapa de Arquitectura (Nexus Cartographer)...
python -W ignore tools/map_project.py

echo [3/5] Analizando Integridad y Conexiones...
python -W ignore nexus_auditor.py

if %ERRORLEVEL% NEQ 0 goto :HALT

echo [OK] Auditoria aprobada. Integridad Verificada.

REM --- FASE 3: LANZAMIENTO ---
echo [3/4] Iniciando Servidor MCP (Feat Sniper)...
echo        - Transporte: SSE (Daemon Persistence / Port 8080)
echo        - Core: ZMQ Autonomous Loop (Active)
echo        - Cerebro: Local Hybrid Model
echo.
echo [4/4] SISTEMA ONLINE. 
echo        [INFO] El bot esta operando en background (ZMQ).
echo        [INFO] El servidor escucha peticiones SSE en http://127.0.0.1:8080
echo ==================================================

REM --- FASE 4: DOCKER / LOCAL HYBRID ---
REM Si se requiere Docker, usar docker-compose up. Este script es para ejecucion LOCAL.
REM No redirigimos stderr para poder ver los logs de Uvicorn (Server Started).
python -W ignore mcp_server.py
goto :END

:HALT
echo.
echo [SYSTEM HALT] La auditoria fallo. Revisa los errores arriba.
pause
exit /b 1

:END
exit /b 0
