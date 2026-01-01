FROM python:3.11-slim
WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/

RUN pip install -U pip && pip install -e .

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
