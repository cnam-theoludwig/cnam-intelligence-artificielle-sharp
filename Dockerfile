FROM ghcr.io/astral-sh/uv:0.11.17-python3.12-trixie-slim@sha256:85477adebb1a01a0d3329ff7068ed2a2773ecf77e81673571d2773a447634ead

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY web ./web

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "src.api"]
