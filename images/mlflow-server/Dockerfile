FROM conda/miniconda3:latest

ENV POSTGRES_USER=mlflow_user
ENV POSTGRES_PASSWORD=mlflow_pass
ENV POSTGRES_DB=mlflow-db
ENV SERVER_PORT=5000
ENV SERVER_HOST=0.0.0.0
ENV DB_URI=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}
ENV AWS_BUCKET_NAME=mlflow
ENV S3_HOST=s3
ENV S3_PORT=9000
ENV MLFLOW_S3_ENDPOINT_URL=http://${S3_HOST}:${S3_PORT}

ENV MLFLOW_HOME=/opt/mlflow
ENV ARTIFACT_STORE=s3://${AWS_BUCKET_NAME}/

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    wget \
    netcat

RUN apt-get -y install python-pip

RUN pip install --upgrade pip

RUN pip install mlflow sqlalchemy psycopg2-binary && \
    mkdir -p ${MLFLOW_HOME}/scripts && \
    #mkdir -p ${FILE_STORE} && \
    mkdir -p ${ARTIFACT_STORE}

# Install minio(S3) client
RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc
RUN mv mc /usr/bin/mc
RUN chmod +x /usr/bin/mc

COPY ./start.sh ${MLFLOW_HOME}/script/start.sh

RUN chmod +x ${MLFLOW_HOME}/script/start.sh


EXPOSE ${SERVER_PORT}/tcp

#VOLUME ["${MLFLOW_HOME}/scripts/", "${FILE_STORE}", "${ARTIFACT_STORE}"]
VOLUME ["${MLFLOW_HOME}/scripts/", "${ARTIFACT_STORE}"]

WORKDIR ${MLFLOW_HOME}

ENTRYPOINT ["./script/start.sh"]