FROM python:3.12-slim-bookworm
ARG GPU=false
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    FACE_SWAP_HOST=0.0.0.0 FACE_SWAP_CLOUD=1 PORT=7860 \
    FACE_SWAP_MODEL_DIR=/app/models FACE_SWAP_DATA_DIR=/app/data
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$GPU" = "true" ]; then \
       pip uninstall -y onnxruntime && pip install --no-cache-dir 'onnxruntime-gpu[cuda,cudnn]>=1.21,<2'; \
       fi
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/data && chown -R appuser:appuser /app
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/healthz', timeout=4)"
CMD ["python", "app.py", "--no-browser"]
