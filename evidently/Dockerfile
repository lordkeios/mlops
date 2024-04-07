FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY config.yaml .

RUN pip3 install -r requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8877"]
