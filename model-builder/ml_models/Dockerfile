FROM continuumio/miniconda3

ARG conda_file_path
ARG conda_env_name
ARG ml_training_file

ENV conda_file_path=${conda_file_path}
ENV conda_env_name=${conda_env_name}
ENV ml_training_file=${ml_training_file}

COPY model-builder app/model-builder
COPY model_class app/model_class

WORKDIR /app

RUN conda env create -f ${conda_file_path}

# Make RUN commands use the new environment:
RUN echo "conda activate ${conda_env_name}" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Activate the environment, and make sure it's activated:
RUN conda activate ${conda_env_name}

CMD python ${ml_training_file}
