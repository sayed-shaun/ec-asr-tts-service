FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Stub src/ so the pip install layer caches on pyproject.toml, not on code
# changes. Real src/ lands via COPY . . and shadows this at import time.
COPY pyproject.toml .
RUN mkdir -p src && touch src/__init__.py
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
