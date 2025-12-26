@echo off
title MT5 Neural Bridge - Enterprise Server
color 0A

echo ========================================
echo   MT5 Neural Bridge v2.0 - Starting...
echo ========================================
echo.

:: 1. Verificar si Python está en el PATH
where python >nul 2>1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado. Instala Python y agregalo al PATH.
    pause
    exit /b
)

:: 2. Matar cualquier instancia previa en el puerto 8000
echo [1/3] Liberando puerto 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo [INFO] Matando proceso PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

:: 3. Cambiar al directorio
cd /d "%~dp0"

:: 4. Iniciar el servidor
echo [2/3] Verificando entorno...
echo.
echo [3/3] Lanzando Gateway FastAPI...
echo ========================================
echo.

python run.py

:: Si falla, mantener abierto para leer error
if %errorlevel% neq 0 (
    echo.
    echo [FATAL] El servidor se detuvo con error.
    pause
)
