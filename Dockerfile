# Two stages: Node builds the web page, then a slim Python image serves the API
# and the built page. Node is not in the final image.
#
# The trained retriever and the corpus are not copied in; mount them at run time
# (see README, "Running with Docker").

# ---- Stage 1: build the web page ----
FROM node:24.16.0-bookworm-slim AS frontend

WORKDIR /frontend
# Dependencies first, so this layer is reused while only source files change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: run the API ----
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# CPU build of PyTorch: much smaller than the CUDA build. requirements.txt then
# finds torch==2.11.0 already installed.
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml LICENSE ./
COPY src/ src/
# The build folder pip leaves behind is not needed at run time.
RUN pip install --no-deps . && rm -rf build

COPY --from=frontend /frontend/dist frontend/dist

# Run as an unprivileged user. Hugging Face files (the generator, when enabled)
# are cached in a folder that user owns, which can also be mounted.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/.cache/huggingface && chown -R app /app/.cache
USER app
ENV HF_HOME=/app/.cache/huggingface

# Answers are off by default: on CPU the 1.5B generator is slow and needs about
# 6 GB of memory. Set RAG_FOR_PANDAS_GENERATOR=Qwen/Qwen2.5-1.5B-Instruct to enable.
ENV RAG_FOR_PANDAS_GENERATOR=none

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "rag_for_pandas.api:app", "--host", "0.0.0.0", "--port", "8000"]
