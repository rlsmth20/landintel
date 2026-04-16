FROM python:3.11

ENV PYTHONPATH=/app:/app/backend

WORKDIR /app/backend

COPY backend/requirements.runtime.txt ./requirements.runtime.txt

RUN pip install -r requirements.runtime.txt

COPY backend/ /app/backend/
COPY floodscraper/ /app/floodscraper/
COPY config/ /app/config/
COPY data/runtime /app/data/runtime
COPY data/tax_published /app/data/tax_published
COPY frontend/public/data /app/frontend/public/data

CMD ["python", "run.py"]
