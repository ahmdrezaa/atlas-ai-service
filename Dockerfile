# Base image Python
FROM python:3.11-slim

# Environment Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Direktori kerja di dalam container
WORKDIR /app

# Multilingual E5 Small (official ONNX export), pinned revision + sha256.
# Bundled at build time and placed before the project install so code changes
# do not re-download it. The service never downloads models at runtime.
ARG E5_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3
ENV FEEDBACK_EMBEDDING_MODEL_DIR=/opt/models/multilingual-e5-small \
    HF_HUB_OFFLINE=1
RUN python - <<'PY'
import hashlib, os, urllib.request
revision = os.environ["E5_REVISION"]
target = os.environ["FEEDBACK_EMBEDDING_MODEL_DIR"]
files = {
    "model.onnx": "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
}
os.makedirs(target, exist_ok=True)
for name, expected in files.items():
    url = f"https://huggingface.co/intfloat/multilingual-e5-small/resolve/{revision}/onnx/{name}"
    path = os.path.join(target, name)
    urllib.request.urlretrieve(url, path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f"Checksum mismatch for {name}")
    print("verified", name)
PY

# Salin konfigurasi project dan source code
COPY pyproject.toml ./
COPY src/ ./src/

# Install project beserta development dependencies
RUN python -m pip install --no-cache-dir ".[dev]"

# Port Python gRPC Server
EXPOSE 50051

# Menjalankan AI Service
CMD ["python", "-m", "atlas_ai.server"]
