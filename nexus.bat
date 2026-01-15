@echo off
setlocal
chcp 65001 >nul
title FEAT NEXUS - SILENT DAEMON
color 0A

REM --- GLOBAL SILENCE PROTOCOL ---
set PYTHONWARNINGS=ignore
set PYTHONDONTWRITEBYTECODE=1

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
echo [PHASE 0] ZOMBIE KILLER AND CLEANUP
echo ==============================================================
powershell -Command "Get-NetTCPConnection -LocalPort 5555 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
powershell -Command "Get-NetTCPConnection -LocalPort 5556 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
taskkill /F /IM python.exe /T >nul 2>&1
echo [OK] Ports and Memory Flushed.

echo.
echo ==============================================================
echo [PHASE 1] ASSETS SYNC
echo ==============================================================
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
python tools/sync_mql5.py

echo.
echo ==============================================================
echo [PHASE 2] IGNITION (SILENT MODE)
echo ==============================================================
echo.
echo [CORE]     Initializing Neural Engine...
echo [STREAM]   Connecting to Supabase Dashboard...
echo [MT5]      Linking to Terminal...
echo.
echo ----------------------------------------------------------------
echo   MINIMAL HUD - DO NOT CLOSE THIS WINDOW
echo ----------------------------------------------------------------
echo   [CORE: ACTIVE]
echo   [MT5:  CONNECTED]
echo   [ZMQ:  STREAMING]
echo.
echo   See RAW logs at: logs/raw_execution.log
echo   See DASHBOARD at: dashboard.html

REM Create logs dir if not exists
if not exist logs mkdir logs

REM Auto-Open Dashboard
start chrome "%~dp0dashboard.html"

REM HARD IGNITION - Visible Output for Debugging
python mcp_server.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [SYSTEM CRASH] Server process failed.
    echo Check the error message above.
    pause
)

echo.
echo [SYSTEM STOPPED] Server process ended.
pause
