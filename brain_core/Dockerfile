# Container Brain for FEAT Sniper MCP
FROM python:3.11-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libzmq3-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
# Note: Heavy libraries (torch, transformers) are installed here
RUN pip install --no-cache-dir \
    fastmcp \
    pyzmq \
    htttpx \
    anyio \
    pandas \
    scikit-learn \
    optuna \
    watchdog \
    python-dotenv \
    supabase \
    chromadb \
    sentence-transformers \
    langchain \
    langchain-community \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy the application code
COPY . .

# Expose MCP port (if using HTTP) and ZMQ Bridge port
EXPOSE 8000 5555

# Run the server
CMD ["python", "mcp_server.py"]
