FROM python:3.10-slim

RUN apt-get update && apt-get upgrade -y
RUN apt-get install libgomp1
# Needed for scikit-optimise
RUN apt install git -y

COPY requirements.txt .

RUN pip install -r requirements.txt
# We currently need to use the unofficial version of scikit optimise
RUN pip install git+https://github.com/QuentinSoubeyran/scikit-optimize.git@unofficial

RUN prefect config set PREFECT_UI_API_URL="http://prefect_server:4200/api"
RUN prefect config set PREFECT_API_URL="http://prefect_server:4200/api"

RUN prefect config view

WORKDIR /app

ADD ./scripts scripts/
COPY ["./main.py", "./"]

# Start the agent and then run the main script to add work to the queue
ENTRYPOINT sleep 15 && prefect agent start --pool default-agent-pool --work-queue credit_risk_model_queue
