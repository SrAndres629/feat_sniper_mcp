# DOCKER STATUS - FEAT Sniper NEXUS

## 📅 Fecha: 2026-01-10

## ✅ Optimización de Layer Caching

### División de Dependencias

| Archivo | Contenido | Tamaño ~|
|---------|-----------|---------|
| `requirements_heavy.txt` | torch, chromadb, sentence-transformers, xgboost, lightgbm, scikit-learn | ~3GB |
| `requirements.txt` | fastmcp, pandas, numpy, uvicorn, etc. | ~200MB |

### Estructura del Dockerfile (4 Capas)

```dockerfile
# CAPA 1: Sistema (gcc, python3-dev)
# CAPA 2: ML Pesado (CACHEADO - torch, chromadb, xgboost)
# CAPA 3: Dependencias ligeras (pandas, fastmcp)
# CAPA 4: Código fuente (cambia frecuentemente)
```

### Comportamiento de Cache

| Cambio | Capas Reconstruidas | Tiempo |
|--------|---------------------|--------|
| Código (`ml_engine.py`) | Solo Capa 4 | ~5s |
| Deps ligeras (`requirements.txt`) | Capas 3+4 | ~30s |
| Deps pesadas (`requirements_heavy.txt`) | Capas 2+3+4 | ~30min |
| Dockerfile | Todas | ~30min |

---

## 📦 Servicios Docker

| Servicio | Puerto | Imagen |
|----------|--------|--------|
| `mcp-brain` | 8000 (SSE), 5555 (ZMQ) | `feat_sniper_mcp-mcp-brain` |
| `web-dashboard` | 3000 | `feat_sniper_mcp-web-dashboard` |

---

## 🔧 Comandos

```bash
# Rebuild completo (primera vez o cambios en deps pesadas)
docker-compose down --rmi all --volumes
docker-compose up --build -d

# Rebuild rápido (solo código)
docker-compose up --build -d
# Verás: => CACHED [mcp-brain 3/5] RUN pip install -r requirements_heavy.txt
```

---

## ⚡ Validación de Cache

Buscar en output:
```
=> CACHED [mcp-brain 3/5] RUN pip install -r requirements_heavy.txt
```

Esto confirma que PyTorch (~900MB) y CUDA (~1.5GB) NO se reinstalaron.

---

*Generado automáticamente por FEAT Sniper NEXUS*
