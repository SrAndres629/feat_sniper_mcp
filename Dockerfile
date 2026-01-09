FROM python:3.11-slim-bullseye

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
# Using bullseye-slim to avoid OOM in build and keeping it minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libzmq3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8000

# Start the MCP server
CMD ["python", "mcp_server.py"]
