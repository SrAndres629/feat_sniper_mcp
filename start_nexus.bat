@echo off
REM ============================================================
REM  FEAT Sniper NEXUS - One-Click Startup
REM  Starts the Docker brain + Dashboard with SSE + RAG Memory
REM ============================================================

echo.
echo  ███████╗███████╗ █████╗ ████████╗    ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
echo  ██╔════╝██╔════╝██╔══██╗╚══██╔══╝    ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
echo  █████╗  █████╗  ███████║   ██║       ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
echo  ██╔══╝  ██╔══╝  ██╔══██║   ██║       ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
echo  ██║     ███████╗██║  ██║   ██║       ██║ ╚████║███████╗██╔╝ ╚██╗╚██████╔╝███████║
echo  ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝       ╚═╝  ╚═══╝╚══════╝╚═╝   ╚═╝ ╚═════╝ ╚══════╝
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)
echo [OK] Docker is running.

REM Navigate to project directory
cd /d "%~dp0"

REM Build and start containers
echo.
echo [INFO] Building and starting NEXUS brain + dashboard...
echo.
docker compose up --build -d

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start containers. Check Docker logs.
    pause
    exit /b 1
)

REM Wait for startup
echo.
echo [INFO] Waiting 10 seconds for services to initialize...
timeout /t 10 /nobreak >nul

REM Show status
echo.
echo ============================================================
echo  NEXUS STATUS
echo ============================================================
docker ps --filter "name=feat-sniper" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

REM Open dashboard in browser
echo [INFO] Opening dashboard in browser...
start http://localhost:3000

REM Stream logs
echo.
echo [INFO] Streaming brain logs (Ctrl+C to stop)...
echo ============================================================
docker logs feat-sniper-brain -f --tail 50
