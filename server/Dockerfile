FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight PyTorch CPU wheels with retries and timeout, strip unused tests/headers (keep torch/bin for torch_shm_manager)
RUN pip install --default-timeout=1000 --retries=10 torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    rm -rf /usr/local/lib/python3.11/site-packages/torch/test \
           /usr/local/lib/python3.11/site-packages/torch/include && \
    rm -rf /root/.cache/pip

# Install python dependencies with pip cache and purge pycache
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt && \
    find /usr/local/lib/python3.11/site-packages/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Copy source code including monitoring
COPY shared /app/shared
COPY server /app/server
COPY client /app/client
COPY monitoring /app/monitoring
COPY experiments /app/experiments

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080 8000

CMD ["python", "-m", "server.app.server_app"]
