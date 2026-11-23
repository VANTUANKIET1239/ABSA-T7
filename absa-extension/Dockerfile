FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/absa-extension/backend
COPY absa-extension/backend/requirements-docker.txt ./
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY absa-extension/backend/ /app/absa-extension/backend/
COPY reprocessing/preprocess_cleanser_reviews.py reprocessing/seller_reply_rules.py /app/reprocessing/
# Keep Java's repeatedly read resources on Linux's filesystem rather than a Windows bind mount.
COPY vncorenlp_model/VnCoreNLP-1.2.jar /models/vncorenlp/VnCoreNLP-1.2.jar
COPY vncorenlp_model/models/wordsegmenter/vi-vocab vncorenlp_model/models/wordsegmenter/wordsegmenter.rdr /models/vncorenlp/models/wordsegmenter/
RUN groupadd --gid 10001 absa \
    && useradd --uid 10001 --gid absa --create-home absa
USER absa
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "10", "--no-access-log"]
