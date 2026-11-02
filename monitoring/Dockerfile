FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies and purge pycache
COPY monitoring/requirements.txt /app/monitoring/requirements.txt
RUN pip install --no-cache-dir -r /app/monitoring/requirements.txt && \
    find /usr/local/lib/python3.11/site-packages/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Copy source code
COPY shared /app/shared
COPY monitoring /app/monitoring

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --retries=5 --start-period=3s \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "monitoring.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
