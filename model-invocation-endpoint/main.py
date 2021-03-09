from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import PlainTextResponse
#import DataInputModel

from pydantic import BaseModel
from typing import List, Union

class DataInputModel(BaseModel):
    columns: List[str]
    data: List[List[Union[int, float]]]

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
def get_inference(name: str, version: int, data: DataInputModel):
    return mlflow_interface.get_inference(name, version, data)

@app.post("/model/{name}/invocation/best")
def get_inference_from_best_model(name: str, data: DataInputModel):
    return  mlflow_interface.get_inference_from_best_model(name, data)

@app.post("/model/{name}/invocation/latest")
def get_inference_from_latest_model(name: str, data: DataInputModel):
    return  mlflow_interface.get_inference_from_latest_model(name, data)