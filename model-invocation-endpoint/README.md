# Model Invocation Endpoint

This module is responsible for getting model inference and details from the mlflow-server.


## Setting the REST API

The model-invocation-enpoint REST API uses `mlflow-server` to fetch model and for inference. To deploy the  mlflow-server docker stack follow the instruction from [here](https://github.com/RADAR-base/RADAR-Docker/tree/distributed-dcompose/distributed-dcompose-stack/mlflow). To train a model and store it in the mlflow-server, you can follow `linear_regression_remote` example from `model-builder/examples`. To set it up follow [README of linear_regression_remote](../model-builder/examples/linear_regression_remote/README.md).

### Deploying the REST API


#### Setting up the ENV file

Create the `.env` file.

```bash
cp env.template .env
```

Use the same `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` values that have been used in `mlflow-server`

#### Deploy the Docker container

Use these commands to run the REST API at `127.0.0.1:80`

```bash
docker build -t model-invocation-endpoint -f model-invocation-endpoint/Dockerfile .
docker run -d --name model-invocation-endpoint -p 80:80 model-invocation-endpoint
```

The REST API can be viewed at `http://127.0.0.1:80/docs#/default`

#### Deploy without Docker

Follow these commands to deploy without using Docker.

```bash
pip install -r requirements.txt
uvicorn main:app
```

## Endpoints Documentation

- GET [http://127.0.0.1/model/](http://127.0.0.1/model) - Fetch details of all the models available in the `mlflow-server`
- GET [http://127.0.0.1/model/{model-name}](http://127.0.0.1/model/{model-name})- Fetch all the details from `mlflow-server` about an experiment with the name `model-name`.
- GET [http://127.0.0.1/model/{model-name}/{version}](http://127.0.0.1/model/{model-name}/{version})- Fetch all the details from `mlflow-server` about an experiment with the name `model-name` and version `version`.
- POST [http://127.0.0.1/model/{model-name}/{version}/invocation](http://127.0.0.1/model/{model-name}/{version}/invocation) - Get inference from the mentioned `version` of the experiment with the name `model-name`. The request body would contain the input data.
- POST [http://127.0.0.1/model/{model-name}/invocation/best](http://127.0.0.1/model/{model-name}/invocation/best) - Get inference from the best performing version of a model. The request body would contain the input data.
- POST [http://127.0.0.1/model/{model-name}/invocation/latest](http://127.0.0.1/model/{model-name}/invocation/latest) - Get inference from the latest performing version of a model. The request body would contain the input data.
