# ============================================
# Dockerfile - FEAT NEXUS MCP Brain
# Optimized: ~2-3GB instead of 13GB
# ============================================
FROM python:3.11-slim

WORKDIR /app

# System dependencies (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer)
# Install PyTorch CPU first (Separate layer for better caching & lighter image)
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Source code (changes frequently)
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONWARNINGS=ignore

# Create logs directory
RUN mkdir -p logs

# Expose ports
EXPOSE 8000 5555

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start MCP Server
CMD ["python", "mcp_server.py"]
