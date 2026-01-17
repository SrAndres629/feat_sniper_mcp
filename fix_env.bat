@echo off
setlocal
title FEAT NEXUS - SELF HEALING PROTOCOL

echo ==========================================
echo [REPAIR] FEAT NEXUS REPAIR AGENT
echo ==========================================

REM 1. Activate Environment
if exist ".venv\Scripts\activate.bat" (
    echo [ENV] Activating .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] .venv missing. Creating...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

REM 2. Upgrade PIP (Vital for Wheels)
echo [PIP] Upgrading PIP...
python -m pip install --upgrade pip

REM 3. Force Re-Install Requirements
echo [INSTALL] Installing Dependencies (This may take a moment)...
python -m pip install -r requirements.txt --force-reinstall

REM 4. Verify Optuna Specifically
echo [VERIFY] Checking Optuna...
python -c "import optuna; print('[OK] Optuna Verified')"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Optuna install failed.
    pause
    exit /b
)

echo.
echo ==========================================
echo [SUCCESS] Environment Repaired.
echo [BOOT] Restarting Nexus...
echo ==========================================
timeout /t 3

call nexus.bat
