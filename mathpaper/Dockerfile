FROM python:3.12-slim

ARG MODEL_PRECISION=int8

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    MATHPAPER_DATA=/app/data \
    MATHPAPER_PUBLIC_DEMO=1 \
    MATHPAPER_PUBLIC_CHAT=0 \
    MATHPAPER_ORT_THREADS=2 \
    MATHPAPER_MODEL_PRECISION=${MODEL_PRECISION} \
    MATHPAPER_INFERENCE_BATCH_SIZE=1 \
    MATHPAPER_DEFAULT_CANDIDATES=20 \
    MATHPAPER_MAX_CANDIDATES=50 \
    ORT_DISABLE_TELEMETRY=1 \
    FASTMCP_CHECK_FOR_UPDATES=off \
    FASTMCP_TELEMETRY_MODE=off

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --chown=1000:1000 pyproject.toml README.md ./
COPY --chown=1000:1000 src ./src
RUN python -m pip install --no-cache-dir ".[agent]"

COPY --chown=1000:1000 data ./data
COPY --chown=1000:1000 deployment ./deployment
USER 1000

# Public model downloads only. No API key is needed or baked into the image.
RUN python deployment/bootstrap.py && mathpaper models && mathpaper index

EXPOSE 7860
CMD ["python", "deployment/start.py"]
