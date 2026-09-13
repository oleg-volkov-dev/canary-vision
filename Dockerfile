FROM ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 AS uv
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS builder

COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY canary_vision ./canary_vision
COPY scripts ./scripts
ENV CANARY_MODEL_PATH=/opt/models/mobilenet_v3_small-047dcff4.pth
RUN .venv/bin/python -m scripts.prepare_model

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS runtime
LABEL org.opencontainers.image.title="CanaryVision" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/oleg-volkov-dev/canary-vision"
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CANARY_MODEL_PATH=/opt/models/mobilenet_v3_small-047dcff4.pth \
    CANARY_CPU_THREADS=2
RUN useradd --uid 10001 --user-group --create-home --shell /usr/sbin/nologin canary
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/models /opt/models
COPY canary_vision ./canary_vision
COPY evaluation ./evaluation
COPY scripts ./scripts
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"
CMD ["uvicorn", "canary_vision.app:app", "--host", "0.0.0.0", "--port", "8000"]
