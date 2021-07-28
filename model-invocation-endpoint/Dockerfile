FROM python:3.8

COPY model-builder app/model-builder

COPY model_class app/model_class

COPY model-invocation-endpoint app/model-invocation-endpoint

WORKDIR app/model-invocation-endpoint

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]