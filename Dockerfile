FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends antiword libarchive-tools \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN mkdir -p src/pbpk_kb && touch src/pbpk_kb/__init__.py
RUN pip install --no-cache-dir -e .

COPY src ./src
RUN ln -s src/pbpk_kb pbpk_kb
COPY tests ./tests

EXPOSE 8000 8501

CMD ["uvicorn", "pbpk_kb.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
