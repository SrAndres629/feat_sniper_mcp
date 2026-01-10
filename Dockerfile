# FEAT Sniper NEXUS - Docker Image
# Optimized with Layer Caching for ML dependencies

FROM python:3.11-slim

WORKDIR /app

# =============================================================================
# LAYER 1: HEAVY ML DEPENDENCIES (Cached - 3GB+, rarely changes)
# PyTorch CPU + ChromaDB + XGBoost
# =============================================================================
COPY requirements_heavy.txt .
RUN pip install --no-cache-dir -r requirements_heavy.txt

# =============================================================================
# LAYER 2: LIGHT DEPENDENCIES (Changes occasionally)
# =============================================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# LAYER 3: SOURCE CODE (Changes frequently)
# This layer rebuilds quickly because previous layers are cached
# =============================================================================
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models

# Expose SSE and ZMQ ports
EXPOSE 8000 5555

# Start MCP Server
CMD ["python", "mcp_server.py"]
