from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import PlainTextResponse
from dataloader_module import DataInputModel, DataLoaderClass

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

@app.post("/model/{name}/best/invocation/")
def get_inference_from_best_model(name: str, data: DataInputModel):
    return  mlflow_interface.get_inference_from_best_model(name, data)


@app.post("/model/{name}/latest/invocation/")
def get_inference_from_latest_model(name: str, data: DataInputModel):
    return  mlflow_interface.get_inference_from_latest_model(name, data)

@app.post("/model/{name}/{version}/invocation")
def get_inference(name: str, version: int, data: DataInputModel):
    return mlflow_interface.get_inference(name, version, data)

@app.post("/model/{name}/latest/metadata-invocation/")
def get_inference_from_latest_model_with_metadata(name: str, metadata: DataLoaderClass, upload: Optional[bool] = True):
    return mlflow_interface.get_inference_from_latest_model_with_metadata(name, metadata, upload)

@app.post("/model/{name}/best/metadata-invocation/")
def get_inference_from_best_model_with_metadata(name: str, metadata: DataLoaderClass, upload: Optional[bool] = True):
    return mlflow_interface.get_inference_from_best_model_with_metadata(name, metadata, upload)

@app.post("/model/{name}/{version}/metadata-invocation/")
def get_inference_with_metadata(name: str, version: int, metadata: DataLoaderClass, upload: Optional[bool] = True):
    return mlflow_interface.get_inference_with_metadata(name, version, metadata, upload)