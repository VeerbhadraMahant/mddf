# syntax=docker/dockerfile:1

# ---- Stage 1: build the React SPA -------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build          # -> /web/dist

# ---- Stage 2: Torch-free Python runtime -----------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    MDDF_LOG_JSON=true

# libGL/libglib for opencv-python-headless runtime deps are already avoided by
# the -headless build; only need a couple of shared libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Install only the runtime dependency set (no PyTorch / Anomalib).
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install .

COPY --from=web /web/dist ./web/dist

RUN mkdir -p /home/app/.cache/huggingface /app/artifacts && chown -R app:app /app /home/app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/v1/health').status==200 else 1)"

CMD ["python", "-m", "mddf.cli", "serve", "--host", "0.0.0.0", "--port", "7860"]
