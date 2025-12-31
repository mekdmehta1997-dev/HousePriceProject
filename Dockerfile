FROM python:3.10
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py /app/training/train.py

CMD ["python", "/app/training/train.py"]
