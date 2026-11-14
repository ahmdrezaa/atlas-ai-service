# syntax=docker/dockerfile:1
# Targets:
#   prod (default, last stage): runtime deps only, non-root, E5 bundled, healthcheck.
#   dev: adds test/lint tools; used by compose.yaml and the test suite.
ARG PYTHON_IMAGE=python:3.11-slim

# --- E5 embedding model: pinned revision + sha256, never downloaded at runtime ----
FROM ${PYTHON_IMAGE} AS models
ARG E5_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3
ENV FEEDBACK_EMBEDDING_MODEL_DIR=/opt/models/multilingual-e5-small
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

# --- Dependencies from uv.lock (exact versions + hashes) --------------------------
FROM ${PYTHON_IMAGE} AS builder
RUN python -m pip install --no-cache-dir "uv==0.13.6"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /build
COPY pyproject.toml uv.lock ./
# Third-party dependencies first (cached layer), then the project itself.
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

FROM builder AS builder-dev
RUN uv sync --frozen --extra dev --no-editable

# --- Shared runtime settings -------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    FEEDBACK_EMBEDDING_MODEL_DIR=/opt/models/multilingual-e5-small \
    HF_HUB_OFFLINE=1 \
    ORT_DISABLE_TELEMETRY=1
COPY --from=models /opt/models /opt/models
WORKDIR /app
EXPOSE 50051

# --- dev: test and lint tools, source mounted by the test runner -------------------
FROM runtime AS dev
COPY --from=builder-dev /opt/venv /opt/venv
COPY pyproject.toml ./
COPY src/ ./src/
CMD ["python", "-m", "atlas_ai.server"]

# --- prod: runtime only, non-root ----------------------------------------------------
FROM runtime AS prod
COPY --from=builder /opt/venv /opt/venv
RUN groupadd --system --gid 10001 atlas \
    && useradd --system --uid 10001 --gid atlas --no-create-home --shell /usr/sbin/nologin atlas
USER 10001:10001
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "atlas_ai.healthcheck"]
CMD ["python", "-m", "atlas_ai.server"]
