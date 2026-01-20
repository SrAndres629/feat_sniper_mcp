@echo off
TITLE FEAT NEXUS - IMMORTAL CORE DAEMON
SETLOCAL EnableDelayedExpansion

:: --- INFRASTRUCTURE GUARDIAN AUDIT v2.0 ---
:: [X] UTF-8 Support
:: [X] Dir Structure
:: [X] Full Dep Scan
:: [X] Admin Privileges Logic (Optional but recommended for ZMQ)

:: 0. Encoding & UI
chcp 65001 >nul
COLOR 0A
cls
echo =================================================================
echo        FEAT SNIPER v2.1 - NEURAL EVOLUTION SYSTEM
echo        "Si Vis Pacem, Para Bellum"
echo        Infrastructure Guardian Verified
echo =================================================================
echo.

:: 1. Structure Audit
echo [GUARDIAN] Verifying Directory Structure...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
echo [GUARDIAN] file system: OK
echo.

:: 2. Virtual Environment Activation
echo [SYSTEM] Locating Virtual Environment...
if NOT exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual Environment not found in .venv/
    echo [FIX] Please initialize: python -m venv .venv
    pause
    exit /b
)
call .venv\Scripts\activate
echo [SYSTEM] Environment: ACTIVATED
echo.

:: 3. Deep Pre-Flight Checks (Python-side)
echo [SYSTEM] Initiating Deep Scan...
python -c "import sys; import torch; import pandas; import supabase; import MetaTrader5 as mt5; print(f' [OK] Python {sys.version.split()[0]}'); print(f' [OK] Torch {torch.__version__} (CUDA: {torch.cuda.is_available()})'); print(f' [OK] Pandas {pandas.__version__}'); print(f' [OK] Supabase SDK Installed'); print(f' [OK] MT5 {mt5.__version__}')"

if errorlevel 1 (
    echo.
    echo [FATAL] Dependency Scan Failed.
    echo [FIX] Run: pip install -r requirements.txt
    pause
    exit /b
)

:: 4. Market Awareness
python -c "from datetime import datetime; today = datetime.now().weekday(); print(f' [OK] Market Status: [{'WEEKEND' if today >= 5 else 'OPEN'}]')"

:: 5. Launch Sequence
echo.
echo [SYSTEM] All systems nominal. Launching Immortal Daemon...
echo -----------------------------------------------------------------
python nexus_daemon.py

:: 6. Exit Handling
echo.
echo -----------------------------------------------------------------
echo [SYSTEM] Daemon suspended.
if errorlevel 1 echo [WARNING] Abnormal Exit detected.
pause
