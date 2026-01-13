@echo off
REM ======================================================
REM DOCKER CLEANUP SCRIPT - Limpia todo lo viejo
REM ======================================================
echo.
echo ================================================
echo   DOCKER CLEANUP - Limpiando sistema Docker
echo ================================================
echo.

REM 1. Detener todos los contenedores
echo [1/5] Deteniendo todos los contenedores...
docker stop $(docker ps -aq) 2>nul
echo [OK]

REM 2. Eliminar todos los contenedores
echo [2/5] Eliminando todos los contenedores...
docker rm -f $(docker ps -aq) 2>nul
echo [OK]

REM 3. Eliminar imagenes huerfanas y sin tag
echo [3/5] Eliminando imagenes huerfanas...
docker image prune -f
echo [OK]

REM 4. Eliminar volumenes no usados
echo [4/5] Eliminando volumenes huerfanos...
docker volume prune -f
echo [OK]

REM 5. Eliminar redes no usadas
echo [5/5] Eliminando redes huerfanas...
docker network prune -f
echo [OK]

echo.
echo ================================================
echo   LIMPIEZA COMPLETADA
echo ================================================
echo.

REM Mostrar estado actual
echo CONTENEDORES ACTIVOS:
docker ps
echo.
echo IMAGENES:
docker images
echo.
echo VOLUMENES:
docker volume ls
echo.

pause
