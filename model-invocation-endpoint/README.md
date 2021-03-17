# Model Invocation Endpoint

This module is responsible for getting model inference and details from the mlflow-server.

## Run the Docker container

Use these commands to run the REST API at `127.0.0.1:80`

```bash
cd model-invocation-endpoint
docker build -t model-invocation-endpoint .
docker run -d --name model-invocation-endpoint -p 80:80 model-invocation-endpoint
```

The REST API can be viewed at `http://127.0.0.1:80/docs#/default`
