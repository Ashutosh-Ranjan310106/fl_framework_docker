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

# Install python dependencies and purge pycache
COPY client/requirements.txt /app/client/requirements.txt
RUN pip install --no-cache-dir -r /app/client/requirements.txt && \
    find /usr/local/lib/python3.11/site-packages/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Copy source code
COPY shared /app/shared
COPY client /app/client

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Volume for client partition
VOLUME /app/data

ENTRYPOINT ["python", "-m", "client.app.client_app"]
CMD []
