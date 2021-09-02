FROM continuumio/miniconda3

ENV conda_file_path=
ENV conda_env_name=
ENV ml_training_file=

COPY model-builder app/model-builder
COPY model_class app/model_class

RUN chmod +x app/model-builder/ml_models/init.sh

WORKDIR /app

CMD ["model-builder/ml_models/init.sh"]