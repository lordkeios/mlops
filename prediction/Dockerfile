FROM python:3.10-slim

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN apt update
RUN apt install -y libgomp1

WORKDIR /app

COPY app.py .

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app:server"]
