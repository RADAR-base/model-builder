#!/bin/bash

# Create the S3 bucket
while ! nc -z ${S3_HOST} ${S3_PORT}; do echo 'Wait minio to startup...' && sleep 0.1; done; \
sleep 5 && \
mc config host add myminio ${MLFLOW_S3_ENDPOINT_URL} ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY} && \
mc mb myminio/$AWS_BUCKET_NAME
mc ls myminio
echo "Created bucket ${AWS_BUCKET_NAME}"

# uncomment below when you see any issue with db...
#mlflow db upgrade $DB_URI

mlflow server \
    --backend-store-uri $DB_URI \
    --default-artifact-root $ARTIFACT_STORE \
    --host $SERVER_HOST \
    --port $SERVER_PORT