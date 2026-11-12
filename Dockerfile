# Base image Python
FROM python:3.11-slim

# Environment Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Direktori kerja di dalam container
WORKDIR /app

# Salin konfigurasi project dan source code
COPY pyproject.toml ./
COPY src/ ./src/

# Install project beserta development dependencies
RUN python -m pip install --no-cache-dir ".[dev]"

# Port Python gRPC Server
EXPOSE 50051

# Menjalankan AI Service
CMD ["python", "-m", "atlas_ai.server"]
