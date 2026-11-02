FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight PyTorch CPU wheels and strip unused tests/headers in the same layer
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /usr/local/lib/python3.11/site-packages/torch/test \
           /usr/local/lib/python3.11/site-packages/torch/include \
           /usr/local/lib/python3.11/site-packages/torch/bin

# Install python dependencies with pip cache and purge pycache
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt && \
    find /usr/local/lib/python3.11/site-packages/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Copy source code
COPY shared /app/shared
COPY server /app/server
COPY client /app/client
COPY experiments /app/experiments

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "-m", "server.app.server_app"]
