# Serverless-style endpoint in a container (closer to the OpenFaaS option). NOT built/tested in the authoring sandbox (no Docker there).
# Build:  docker build -t churn-endpoint .
# Run:    docker run -p 8080:8080 -v "$PWD/mlflow.db:/app/mlflow.db" -v "$PWD/mlartifacts:/app/mlartifacts" churn-endpoint
# The registry (SQLite + artifacts) is mounted, so promoting a model in the registry changes what the container serves - no rebuild.
FROM python:3.12-slim
WORKDIR /app
COPY function/requirements.txt function/requirements.txt
RUN pip install --no-cache-dir -r function/requirements.txt scipy
COPY src src
COPY function function
COPY configs configs
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db REFRESH_SECONDS=2
EXPOSE 8080
CMD ["functions-framework", "--source", "function/main.py", "--target", "predict", "--port", "8080", "--host", "0.0.0.0"]
