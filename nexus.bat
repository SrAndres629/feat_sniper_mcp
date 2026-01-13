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
REM --- FASE 1.5: DOCKER BRAIN BOOT ---
echo [1.5/5] Iniciando Cerebro Neural (Docker)...
docker-compose up -d
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Docker no inicio correctamente. Continuando en modo degradado...
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
echo        - Transporte: STDIO
echo        - Logging: logs/nexus_system.log
echo.
echo [4/4] SISTEMA ONLINE. Esperando conexion de VS Code...
echo ==================================================

REM Redirigir stderr a archivo para silenciar warnings de pyiceberg y banner de FastMCP
python -W ignore mcp_server.py 2>logs\mcp_warnings.log
goto :END

:HALT
echo.
echo [SYSTEM HALT] La auditoria fallo. Revisa los errores arriba.
exit /b 1

:END
exit /b 0
