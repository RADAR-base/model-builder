FROM continuumio/miniconda3

COPY model-builder app/model-builder
COPY model_class app/model_class

WORKDIR /app

RUN conda env create -f model-builder/ml_models/anamoly_detection/lung_study/conda.yaml

# Make RUN commands use the new environment:
RUN echo "conda activate lung_study_lstm_anomaly_detection" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Activate the environment, and make sure it's activated:
RUN conda activate lung_study_lstm_anomaly_detection

CMD python model-builder/ml_models/anamoly_detection/lung_study/train.py
