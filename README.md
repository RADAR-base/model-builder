# Model Invocation Endpoint

This module is responsible for getting model inference and details from the mlflow-server. This REST module will run inference on the trained model and will require sufficient amount of memory to load a model and run inference.

## Set Up

To set up the rest invocation api, copy .env file using:

```bash
cd model-invocation-endpoint
cp env.template .env
```

Updated the variables in the `.env` file.
## Run the Docker container

Use these commands to run the REST API at `127.0.0.1:80`

```bash
docker build -t model-invocation-endpoint -f model-invocation-endpoint/Dockerfile
docker run -d --name model-invocation-endpoint -p 80:80 model-invocation-endpoint
```

The REST API can be viewed at `http://127.0.0.1:80/docs#/default`

## Rest API docs

There are two different ways to get inference using the REST API

### 1. Using Raw Data

Raw data can be uploaded directly using `invocation' endpoint. You
To get inference from a specific version of a specific model, we can use:

```bash
curl -X 'POST' \
  'http://0.0.0.0/model/{model_name}/{model_verson}/invocation' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "columns": [
    "string"
  ],
  "data": {raw_data},
  "format": {format of raw_data}
}'
```

Currently supported formats are [pandas_split, pandas_record, tf-instances and tf-inputs](./docs/input_formats.md).

To get inference from the best performing model, we can use:
```bash
curl -X 'POST' \
  'http://0.0.0.0/model/{model_name}/invocation/best' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "columns": [
    "string"
  ],
  "data": {raw_data},
  "format": {format of raw_data}
}'
```

To get inference from the latest trained model, we can use:
```bash
curl -X 'POST' \
  'http://0.0.0.0/model/{model_name}/invocation/latest' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "columns": [
    "string"
  ],
  "data": {raw_data},
  "format": {format of raw_data}
}'
```

### 2. Using Data Location

Another method to get inference is to give location of the data in the postgres db. The api would fetch the data from the database and return the inference results.

To get inference from a specific version of a specific model, we can use:

```bash
curl -X 'POST' \
  'http://0.0.0.0/model/{model_name}/{model_version}/metadata-invocation/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "filename": "string",
  "classname": "string",
  "dbname": "string",
  "starttime": "2021-07-28T14:16:43.026Z",
  "endtime": "2021-07-28T14:16:43.026Z",
  "user_id": "string"
}'
```

Here `filename` and `classname` will be used to identify the study. The object of class `classname` from file `filename` is supposed generate queries which will be used to fetch data from the postgres db and the `preprocess_data` function from class `classname` is used to pre process the data.

In the case of lung study, the classname is `LungStudy` and filename is `lung_study`. This will fetch `LungStudy` class from the `model_class/lung_study.py` file.

`dbname` is the name of the database.

The desired data from user with id `user_id` will be fetched between time `starttime` and `endtime`. `starttime` and `endtime` are optional parameters and can be None if the desired data do not have any `starttime` or `endtime`.


To get inference from the best performing model, we can use:

```bash
curl -X 'POST' \
  'http://0.0.0.0/model/{model_name}/metadata-invocation/best' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "filename": "string",
  "classname": "string",
  "dbname": "string",
  "starttime": "2021-07-28T14:16:43.026Z",
  "endtime": "2021-07-28T14:16:43.026Z",
  "user_id": "string"
}'
```

To get inference from the latest trained model, we can use:
```bash
curl -X 'POST' \
  'http://0.0.0.0/model/{model_name}/metadata-invocation/latest' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "filename": "string",
  "classname": "string",
  "dbname": "string",
  "starttime": "2021-07-28T14:16:43.026Z",
  "endtime": "2021-07-28T14:16:43.026Z",
  "user_id": "string"
}'
```