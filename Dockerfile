FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY src ./src
COPY api ./api
COPY configs ./configs
COPY scripts ./scripts
COPY demo ./demo

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src:/app
ENV BONEMEDVQA_CONFIG=/app/configs/lightweight.yaml

EXPOSE 8000 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
