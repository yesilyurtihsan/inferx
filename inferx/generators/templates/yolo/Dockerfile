# Multi-stage Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir -e .

FROM python:3.11-slim as runtime

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

EXPOSE 8080

CMD ["python", "-m", "src.server"]