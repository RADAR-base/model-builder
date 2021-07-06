FROM python:3.8

COPY . ./model-builder/app

COPY ../model-builder ./model-builder/model-builder

COPY ../model_class ./model-builder/model_class

WORKDIR model-builder/app

RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]