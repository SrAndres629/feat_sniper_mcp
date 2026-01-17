@echo off
setlocal
chcp 65001 >nul
title FEAT NEXUS - SILENT DAEMON
color 0A

REM --- GLOBAL SILENCE PROTOCOL ---
set PYTHONWARNINGS=ignore
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8

cls
echo.
echo  ███████╗███████╗ █████╗ ████████╗    ███████╗███╗   ██╗██╗██████╗ ███████╗██████╗ 
echo  ██╔════╝██╔════╝██╔══██╗╚══██╔══╝    ██╔════╝████╗  ██║██║██╔══██╗██╔════╝██╔══██╗
echo  █████╗  █████╗  ███████║   ██║       ███████╗██╔██╗ ██║██║██████╔╝█████╗  ██████╔╝
echo  ██╔══╝  ██╔══╝  ██╔══██║   ██║       ╚════██║██║╚██╗██║██║██╔═══╝ ██╔══╝  ██╔══██╗
echo  ██║     ███████╗██║  ██║   ██║       ███████║██║ ╚████║██║██║     ███████╗██║  ██║
echo  ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝       ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝                                                                                
echo.
echo      [ SILENT DAEMON NODE ]
echo      [ STATUS: ACTIVE ^| MODE: BACKGROUND EXECUTION ]
echo.
echo ==============================================================
echo [PHASE 0] ZOMBIE KILLER (SYSTEM PURGE)
echo ==============================================================
echo [SANITIZER] Scanning ports 5555-5558, 8000...
powershell -NoProfile -Command "5555,5556,5557,5558,8000 | ForEach-Object { $p=$_; Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue | ForEach-Object { Write-Host 'Killing PID:' $_.OwningProcess 'on port' $p; Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } }"
echo [SANITIZER] Flushing Python Orphans...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq FEAT NEXUS*" >nul 2>&1
echo [OK] System Clean.

echo.
REM [PHASE 1] ASSETS SYNC
echo ==============================================================
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual Environment not found. Proceeding with global python...
)

echo [SYNC] Mapping MQL5 Assets...
python tools/sync_mql5.py

REM [PHASE 1.5] PRE-FLIGHT CHECKS
echo ==============================================================
echo [CHECK] Verifying Neural Manifest...
tasklist /FI "IMAGENAME eq terminal64.exe" | findstr /I "terminal64.exe" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MT5 Terminal not detected. Please start LiteFinance MT5.
    pause
    exit /b
)
echo [OK] MT5 Terminal found.
echo [CHECK] Verifying Neural Hardware (Torch CUDA)...
python -c "import torch; print('[OK] CUDA Available:' if torch.cuda.is_available() else '[WARNING] Running on CPU (Institutional Standards recommend GPU)')"
echo [CHECK] Verifying JIT Acceleration (Numba)...
python -c "import numba; print('[OK] Numba JIT Active.')"
echo [CHECK] Verifying Supabase Connectivity...
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri '%SUPABASE_URL%' -Method Head -TimeoutSec 5; Write-Host '[OK] Supabase Reachable.' } catch { Write-Host '[ERROR] Supabase Offline or URL Missing.' -ForegroundColor Red; exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo [CRITICAL] Cloud Connectivity failure. System cannot sync trade logs.
    pause
    exit /b
)
echo [CHECK] Verifying ZMQ Ports Availability (5555-5558)...
powershell -NoProfile -Command "5555,5556,5557,5558 | ForEach-Object { $p=$_; if (Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue) { Write-Host '[ERROR] Port' $p 'is BUSY' -ForegroundColor Red; exit 1 } }; Write-Host '[OK] ZMQ Ports Clear.'"
if %ERRORLEVEL% NEQ 0 (
    echo [CRITICAL] Port conflict detected. Another instance might be running.
    pause
    exit /b
)
echo ----------------------------------------------------------------
echo.
echo [CORE]     Initializing Neural Intelligence...
echo [CORTEX]   Activating Visual Feedback Layer...
echo [MT5]      Syncing Terminal Context...
echo.
echo ----------------------------------------------------------------
echo   FEAT NEXUS: OPERACION SINGULARITY - OPERATIONAL
echo ----------------------------------------------------------------
echo   [NEURAL_CORE:  ACTIVE]
echo   [VISUAL_CORTEX: STARTING]
echo   [ZMQ_STREAM:    BUSY]
echo.
echo   Note: Closing this window will terminate all subsystems.
echo.

REM Create logs dir if not exists
if not exist logs mkdir logs

REM [LEVEL 53] Unified Dashboard Activation
REM Launches Streamlit in a separate asynchronous terminal thread
start "VISUAL CORTEX" /min cmd /c "streamlit run app/dashboard/neural_viz.py"

REM [CORE] MAIN EXECUTION BLOCK (Sync)
python mcp_server.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [SYSTEM CRASH] Neural Core failure detected.
    echo Protocol 66: Monitoring Logs...
    pause
)

echo.
echo [SYSTEM STOPPED] Core Process Ended.

echo [CLEANUP] Terminal De-sync in progress...
REM Kill Streamlit and all relevant python ghosts
taskkill /F /IM python.exe /FI "WINDOWTITLE eq VISUAL CORTEX*" >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq FEAT NEXUS*" >nul 2>&1
echo [CLEANUP] All Nodes Terminated.

echo.
echo Done.
pause
