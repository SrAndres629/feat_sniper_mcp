# FEAT Sniper NEXUS - Docker Image
# Optimized with Layer Caching for ML dependencies

FROM python:3.11-slim

WORKDIR /app

# =============================================================================
# LAYER 1: DEPENDENCIES (Consolidated)
# =============================================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# LAYER 3: SOURCE CODE (Changes frequently)
# This layer rebuilds quickly because previous layers are cached
# =============================================================================
COPY . .

# Create a non-root user to run the app
# RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories and set ownership
# RUN mkdir -p /app/data /app/models && chown -R appuser:appuser /app

# Switch to non-root user
# USER appuser

# Expose SSE and ZMQ ports
EXPOSE 8000 5555

# Start MCP Server
CMD ["python", "mcp_server.py"]
