FROM python:3.11

WORKDIR /app/backend

COPY backend/requirements.runtime.txt ./requirements.runtime.txt

RUN pip install -r requirements.runtime.txt

COPY backend/ ./
COPY frontend/public/data /app/frontend/public/data
COPY data/tax_published/ms /app/data/tax_published/ms

CMD ["python", "run.py"]
