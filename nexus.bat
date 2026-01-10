@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title NEXUS INSTITUTIONAL COMMAND CENTER - MASTER AUDITOR
color 0A

:: ============================================================
:: CONFIGURACIÓN VISUAL Y VARIABLES
:: ============================================================
set "S_LINE=------------------------------------------------------------"
set "D_LINE============================================================="
set "PREFIX=[NEXUS-CORE]"

:: Puertos Clave
set "PORT_ZMQ=5555"
set "PORT_API=8000"
set "PORT_WEB=3000"

:: Rutas MT5
set "MT5_PATH_1=C:\Program Files\LiteFinance MT5 Terminal\terminal64.exe"
set "MT5_PATH_2=C:\Program Files\MetaTrader 5\terminal64.exe"

cls
echo.
echo %D_LINE%
echo   FEAT SNIPER NEXUS - TOPOLOGIA DE SISTEMA EN TIEMPO REAL
echo %D_LINE%
echo.

:: ============================================================
:: [CAPA 1] VERIFICACIÓN FÍSICA (ARCHIVOS Y ENTORNO)
:: ============================================================
echo %PREFIX% AUDITANDO INTEGRIDAD DE ARCHIVOS (INPUTS)...

:: CHECK .ENV
if exist ".env" goto ENV_OK
color 0C
echo   [-] Archivo .env ................. [MISSING]
echo       CRITICO: Falta configuracion de credenciales.
echo       Por favor cree el archivo .env con SUPABASE_URL y SUPABASE_KEY.
pause
exit /b

:ENV_OK
echo   [+] Archivo .env ................. [OK] (Configuracion cargada)

:: CHECK AUDITOR
if exist "nexus_auditor.py" goto AUDITOR_OK
echo   [-] Auditor Logico ............... [MISSING]
echo       CRITICO: No se encuentra nexus_auditor.py
pause
exit /b

:AUDITOR_OK
echo   [+] Auditor Logico (Python) ...... [OK]

:: CHECK MODELS (Flexible path)
if exist "app\models\gbm_v1.joblib" goto MODEL_APP_OK
if exist "models\gbm_v1.joblib" goto MODEL_ROOT_OK
echo   [!] Modelo ML (GBM) .............. [PENDING] (Se entrenara al iniciar)
goto LAYER_2

:MODEL_APP_OK
echo   [+] Modelo ML (GBM) .............. [OK] (En app/models)
goto LAYER_2

:MODEL_ROOT_OK
echo   [+] Modelo ML (GBM) .............. [OK] (En models/)

:LAYER_2
echo %S_LINE%

:: ============================================================
:: [CAPA 2] VERIFICACIÓN DE PROCESOS (MOTORES)
:: ============================================================
echo.
echo %PREFIX% AUDITANDO MOTORES DE EJECUCION...

:: 1. DOCKER CHECK
docker info >nul 2>&1
if %errorlevel% equ 0 goto DOCKER_ONLINE
echo   [MOTOR] Docker Engine ............ [OFFLINE] - CRITICO
echo           Iniciando Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo           Esperando a Docker (15s)...
timeout /t 15 >nul
:: Re-check after wait
docker info >nul 2>&1
if %errorlevel% neq 0 echo   [ALERTA] Docker aun no responde.

:DOCKER_ONLINE
echo   [MOTOR] Docker Engine ............ [ONLINE]
:: Check Brain Container
docker ps | find "feat-sniper-brain" >nul
if %errorlevel% equ 0 goto BRAIN_RUNNING
echo   [CONTENEDOR] Brain API ........... [STOPPED] - Iniciando...
docker compose up -d
timeout /t 5 >nul
goto MT5_CHECK

:BRAIN_RUNNING
echo   [CONTENEDOR] Brain API ........... [RUNNING]

:MT5_CHECK
:: 2. MT5 CHECK
tasklist /FI "IMAGENAME eq terminal64.exe" 2>NUL | find /I /N "terminal64.exe">NUL
if "%ERRORLEVEL%"=="0" goto MT5_RUNNING

echo   [MOTOR] MetaTrader 5 ............. [OFFLINE]
echo           Lanzando Terminal Institucional...

if exist "%MT5_PATH_1%" (
    start "" "%MT5_PATH_1%"
    echo           Iniciando LiteFinance MT5...
    timeout /t 10 >nul
    goto MT5_CHECK_DONE
)

if exist "%MT5_PATH_2%" (
    start "" "%MT5_PATH_2%"
    echo           Iniciando MetaTrader 5 Estandar...
    timeout /t 10 >nul
    goto MT5_CHECK_DONE
)
echo           [ALERTA] No se pudo encontrar terminal64.exe
goto MT5_CHECK_DONE

:MT5_RUNNING
echo   [MOTOR] MetaTrader 5 ............. [ONLINE]

:MT5_CHECK_DONE
echo %S_LINE%

:: ============================================================
:: [CAPA 3] VISUALIZACIÓN DE INTERACCIONES (PUENTES)
:: ============================================================
echo.
echo %PREFIX% MAPA DE INTERACCIONES (ENTRADAS <-> SALIDAS)...
echo.

:: CHECK ZMQ PORT
netstat -an | find "%PORT_ZMQ%" | find "LISTENING" >nul
if %errorlevel% equ 0 (
    echo   [MERCADO] --(Ticks/Precios^) --^> [PUENTE ZMQ:5555] --(Data^) --^> [PYTHON CORE]
    echo                                         ^^^|
    echo                                     ESTADO: CONECTADO [OK]
) else (
    echo   [MERCADO] --X-- [PUENTE ZMQ:5555] --X-- [PYTHON CORE]
    echo                 ESTADO: DESCONECTADO (Esperando MT5...)
)

echo.

:: CHECK API PORT
netstat -an | find "%PORT_API%" | find "LISTENING" >nul
if %errorlevel% equ 0 (
    echo   [CEREBRO] --(Analisis ML^) ----^> [API REST:8000] --(JSON^) --^> [DASHBOARD]
    echo                                         ^^^|
    echo                                     ESTADO: ACTIVO [OK]
) else (
    echo   [CEREBRO] --X-- [API REST:8000] --X-- [DASHBOARD]
    echo                 ESTADO: INACTIVO (Revisar Docker logs)
)

:LAYER_4
echo %S_LINE%

:: ============================================================
:: [CAPA 4] AUDITORIA PROFUNDA DE DATOS (LOGICA INTERNA)
:: ============================================================
echo.
echo %PREFIX% INICIANDO SONDA DE PROFUNDIDAD (PYTHON AUDITOR)...
echo.
python nexus_auditor.py

if %errorlevel% neq 0 (
    echo.
    echo   [!] EL AUDITOR REPORTÓ ANOMALIAS CRITICAS.
    echo       Revisa los mensajes en ROJO arriba.
    
    choice /C RS /N /M "[R] Run Auto-Healer (Start System)  [S] Stop : "
    if errorlevel 2 goto END
    if errorlevel 1 goto LAUNCH
)

echo.
echo %D_LINE%
echo   ESTADO FINAL DEL SISTEMA: OPERATIVO
echo %D_LINE%
echo.

:: ============================================================
:: [CAPA 5] LANZAMIENTO
:: ============================================================

choice /C SN /N /M "¿Deseas lanzar el Dashboard y el Monitor de Logs ahora? [S/N] "
if errorlevel 2 goto END
if errorlevel 1 goto LAUNCH

:LAUNCH
echo.
echo   [LAUNCH] Abriendo Dashboard...
start http://localhost:3000

echo   [LAUNCH] Iniciando Nexus Control Loop...
python nexus_control.py start
goto END

:END
echo.
echo   [NEXUS] Sesion finalizada.
pause
