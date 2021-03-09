from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from mlflow_interface import MlflowInterface
import mlflow

app = FastAPI()
mlflow_interface = MlflowInterface()

@app.get("/model")
def get_all_experiments():
    return mlflow_interface.get_all_experiments()


@app.get("/model/{name}")
def get_model_info(name: str):
    return mlflow_interface.get_model_info(name)

@app.get("/model/{name}/{version}")
def get_model_version_info(name: str, version: int):
    return mlflow_interface.get_model_version_info(name, version)

#@app.post("/model/{name}")
#def train_new_version():
#    return 0

@app.post("/model/{name}/{version}/invocation")
def get_inference(name: str, version: int):
    return {"Hello": "World"}

@app.post("/model/{name}/best/invocation")
def get_inference_from_best_model(name: str):
    return {"Hello": "World"}

@app.post("/model/{name}/latest/invocation")
def get_inference_from_latest_model(name: str):
    return {"Hello": "World"}