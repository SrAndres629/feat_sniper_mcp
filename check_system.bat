@echo off
REM ============================================================
REM FEAT SNIPER NEXUS - System Status Checker
REM ============================================================
REM Double-click this file to check if all systems are running.
REM ============================================================

echo.
echo   ====================================================
echo            FEAT SNIPER NEXUS - SYSTEM CHECK
echo   ====================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker no esta corriendo. Inicia Docker Desktop primero.
    pause
    exit /b 1
)

REM Check if container exists
docker ps -q -f name=feat-sniper-brain >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Contenedor feat-sniper-brain no encontrado.
    echo         Ejecuta start_nexus.bat primero.
    pause
    exit /b 1
)

REM Run diagnostic inside container
docker exec -it feat-sniper-brain python /app/nexus_status.py

echo.
echo   Presiona cualquier tecla para cerrar...
pause >nul
