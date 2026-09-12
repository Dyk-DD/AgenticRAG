# Agentic-RAG API + SPA 镜像

FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app


ARG TORCH_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --index-url ${TORCH_INDEX_URL} --retries 10 --timeout 120 \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0

# ---- 其余 Python 依赖 ------------------------------------------------
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirements.txt /app/requirements.txt
RUN grep -viE '^torch' /app/requirements.txt > /tmp/requirements-api.txt \
    && pip install --index-url ${PIP_INDEX_URL} --retries 10 --timeout 120 \
       -r /tmp/requirements-api.txt \
    && rm -f /tmp/requirements-api.txt

# ---- 应用代码（改动最频繁，放最后）------------------------------------
COPY app/ /app/app/

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/static \
    && chown -R appuser:appuser /app/data /app/static
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
