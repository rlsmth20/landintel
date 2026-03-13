FROM python:3.11

WORKDIR /app

COPY requirements.runtime.txt .

RUN pip install -r requirements.runtime.txt

COPY . .

CMD ["python", "run.py"]
