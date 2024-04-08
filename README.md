# MLOps - Minio, MLFlow, Prefect

## docker compose
docker compose --env-file config.env up -d --build
docker compose --env-file config.env down
docker compose --env-file config.env build prefect_agent && docker compose --env-file config.env restart prefect_agent

## urls
- Minio: http://localhost:9000/
- Minio UI: http://localhost:9001/
- MLFlow: http://localhost:5051/
- Prefect UI: http://localhost:4200/
- Model UI: http://localhost:9696/
- Prometheus: http://localhost:9091/
- Grafana Dashboard: http://localhost:3000/ 

## commands
docker exec prefect_agent python3 /app/main.py
docker exec reporting_agent python3 /app/generate_evidently_report.py
docker exec prefect_agent prefect config view
docker exec prefect_agent prefect work-pool create default-agent-pool
docker exec prefect_agent ls /app

