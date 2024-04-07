FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl
RUN pip install prefect

RUN mkdir -p /data/store

EXPOSE 4200
EXPOSE 8081

CMD prefect config set PREFECT_UI_API_URL=$LOCAL_PREFECT_ADDRESS && prefect server start --host 0.0.0.0
