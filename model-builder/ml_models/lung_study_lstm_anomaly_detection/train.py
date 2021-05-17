import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
# Get the current working directory
sys.path.append(os.path.abspath('model-builder/'))
from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
from dataloader.querybuilder import QueryBuilder
sys.path.append(os.path.abspath('.'))
from model_class.lung_study import  LungStudy
from lstm import LSTM, LSTMWrapper, LungStudyDataset

from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
import torch
from torch import nn
import time
from utils import fig2data, result_plot

import mlflow.pyfunc

def set_env_vars():
    os.environ["MLFLOW_URL"] = "http://127.0.0.1:5000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = ""
    os.environ["AWS_SECRET_ACCESS_KEY"] = ""


def log_scalar(name, value, step):
    """Log a scalar value to MLflow"""
    mlflow.log_metric(name, value, step=step)

def train_model( train_dataset, val_dataset, model, device, n_epochs, lr):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss(reduction='mean').to(device)
    history = dict(train_loss=[], val_loss=[])
    for epoch in range(1 , n_epochs + 1):
        model = model.train()
        ts = time.time()
        train_losses = []

        for seq_true in train_dataset:
            seq_true = seq_true.to(device)
            y_true = seq_true[:,-1,:]
            optimizer.zero_grad()
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, y_true)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                y_true = seq_true[:,-1,:]
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, y_true)

                val_losses.append(loss.item())

        te = time.time()
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        log_scalar("training_loss", train_loss, step=epoch)
        mlflow.log_metric("validation_loss", val_loss, step=epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch: {epoch}  train loss: {train_loss}  val loss: {val_loss}  time: {te-ts} ")

    return val_loss, history

def train_val_dataset(dataset, dataset_index, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_dataset_idx = Subset(dataset_index, train_idx)
    val_dataset_idx = Subset(dataset_index, val_idx)
    return LungStudyDataset(train_dataset, train_dataset_idx), LungStudyDataset(val_dataset, val_dataset_idx)

def get_device():
    if torch.cuda.is_available():
        return "gpu"
    else:
        return "cpu"

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    set_env_vars()
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", default=5)
    parser.add_argument("--latent_dim", default=128)
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--learning_rate", default=0.01)
    args = parser.parse_args()

    lstm_conda_env={'channels': ['defaults'],
     'name':'lstm_conda_env',
     'dependencies': [ 'python=3.8', 'pip',
     {'pip':['mlflow','pytorch','cloudpickle','pandas','numpy', 'torchvision']}]}

    # Importing query builder
    querybuilder = QueryBuilder(tablename="")
    # Read the wine-quality data from the Postgres database using dataloader
    # ADD DATABASE DETAILS HERE
    lung_study = LungStudy()
    dbconn = PostgresPandasWrapper(dbname="", user="", password="",host="", port=)
    dbconn.connect()
    #print(dbconn.get_response(cols=["*"], dataset="wine_dataset"))
    query = lung_study.get_query_for_training()
    data = dbconn.get_response(query)
    dbconn.disconnect()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #Set experiment
    mlflow.set_experiment("lung_study_lstm_anomaly_detection")

    dataset, dataset_index = lung_study.preprocess_data(data)

    num_layers = args.num_layers
    latent_dim = args.latent_dim
    input_dim = dataset.shape[1]
    input_dimensionality = dataset.shape[2]
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    device =  get_device()
    train_dataset, val_dataset = train_val_dataset(dataset, dataset_index)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    with mlflow.start_run() as run:
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("input_dimensionality", input_dimensionality)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("device", device)

        lstm = LSTM(input_dimensionality, input_dim, latent_dim, num_layers)
        threshold, history = train_model( train_dataloader, val_dataloader, lstm, device, epochs, lr)
        mlflow.log_image(fig2data(result_plot(history)), "result_plot.png")
        mlflow.log_param("Estimated Threshold", threshold)

        mlflow.pyfunc.log_model(artifact_path="lung_study_lstm_anomaly_detection", python_model=LSTMWrapper(model=lstm, threshold=threshold), conda_env=lstm_conda_env, code_path=["model-builder/ml_models/lung_study_lstm_anomaly_detection/lstm.py"])
