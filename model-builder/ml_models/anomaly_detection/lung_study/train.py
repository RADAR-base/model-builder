import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
# Get the current working directory
sys.path.append(os.path.abspath('model-builder/'))
from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
from dataloader.querybuilder import QueryBuilder
sys.path.append(os.path.abspath('.'))
from model_class.lung_study import  LungStudy
from ml_models.anomaly_detection.lstm_attention import RecurrentAutoencoder, LSTMAnomalyDataset
from ml_models.anomaly_detection.lung_study.wrapper import LSTMLungStudyWrapper
from ml_models.anomaly_detection.utils import fig2data, result_plot
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader
import torch
from torch import nn
import time
import copy
import mlflow.pyfunc



def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc_layer", default=3)
    parser.add_argument("--dec_layer", default=3)
    parser.add_argument("--latent_dim", default=128)
    parser.add_argument("--epochs", default=50)
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--learning_rate", default=0.01)
    args = parser.parse_args()
    return args

def get_postgres_data():
        postgres_data = {}
        postgres_data["user"] = os.environ.get('POSTGRES_USER')
        postgres_data["password"] = os.environ.get('POSTGRES_PASS')
        postgres_data["tablename"] = os.environ.get('POSTGRES_TABLENAME')
        postgres_data["host"] = os.environ.get('POSTGRES_HOST')
        postgres_data["port"] = os.environ.get('POSTGRES_PORT')
        postgres_data["dbname"] = os.environ.get('POSTGRES_DBNAME')
        return postgres_data

def get_mlflow_uris():
        return os.environ.get('MLFLOW_TRACKING_URI'), os.environ.get('MLFLOW_S3_ENDPOINT_URL'), os.environ.get('MLFLOW_EXPERIMENT_NAME')

def import_data():
    lung_study = LungStudy()
    postgres_data = get_postgres_data()
    # Importing query builder
    querybuilder = QueryBuilder(tablename=postgres_data["tablename"])
    # Read the wine-quality data from the Postgres database using dataloader
    # ADD DATABASE DETAILS HERE
    dbconn =  PostgresPandasWrapper(dbname=postgres_data["dbname"], user=postgres_data["user"], password=postgres_data["password"],host=postgres_data["host"], port=postgres_data["port"])
    dbconn.connect()
    query = lung_study.get_query_for_training()
    data = dbconn.get_response(query)
    dbconn.disconnect()
    processed_data = lung_study.preprocess_data(data)
    if processed_data is None:
        raise ValueError("Not enough data for training")
    return processed_data

def train_model( train_dataset, val_dataset, model, device, n_epochs, lr):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=n_epochs, steps_per_epoch=len(train_dataset))

    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train_loss=[], val_loss=[])
    best_loss = np.inf
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
            scheduler.step()

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
        train_loss = np.sum(train_losses) / len(train_dataset.dataset)
        val_loss = np.sum(val_losses) / len(val_dataset.dataset)
        mlflow.log_metric("training_loss", train_loss, step=epoch)
        mlflow.log_metric("validation_loss", val_loss, step=epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch: {epoch}  train loss: {train_loss}  val loss: {val_loss}  time: {te-ts} ")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    return best_model_wts, val_loss, history

def train_val_dataset(dataset, dataset_index, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    train_dataset_idx = Subset(dataset_index, train_idx)
    val_dataset_idx = Subset(dataset_index, val_idx)
    return LSTMAnomalyDataset(train_dataset, train_dataset_idx), LSTMAnomalyDataset(val_dataset, val_dataset_idx)

def get_device():
    if torch.cuda.is_available():
        return "gpu"
    else:
        return "cpu"

def main():
    np.random.seed(42)
    args = argparser()
    load_dotenv('model-builder/ml_models/anomaly_detection/lung_study/.env')

    lstm_conda_env={'channels': ['defaults'],
     'name':'lstm_conda_env',
     'dependencies': [ 'python=3.8', 'pip',
     {'pip':['mlflow','torch==1.10.2','cloudpickle','pandas','numpy', 'torchvision']}]}

    mlflow_tracking_uri, mlflow_registry_uri, mlflow_experiment_name = get_mlflow_uris()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    try:
        dataset, dataset_index = import_data()
    except ValueError as e:
        with mlflow.start_run(tags={"alias":"lstmad"}):
            mlflow.set_tag("LOG_STATUS", f"FAILED: {e}")
            sys.exit(1)

    enc_layers = args.enc_layer
    dec_layers = args.dec_layer
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

    with mlflow.start_run(tags={"alias":"lstmad"}) as run:
        mlflow.log_param("enc_layers", enc_layers)
        mlflow.log_param("dec_layers", dec_layers)
        mlflow.log_param("latent_dim", latent_dim)
        mlflow.log_param("input_dim", input_dim)
        mlflow.log_param("input_dimensionality", input_dimensionality)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("device", device)

        lstm = RecurrentAutoencoder(input_dim, input_dimensionality, latent_dim, enc_layers, dec_layers)
        best_model_wts, threshold, history = train_model( train_dataloader, val_dataloader, lstm, device, epochs, lr)
        best_model = RecurrentAutoencoder(input_dim, input_dimensionality, latent_dim, enc_layers, dec_layers)
        best_model.load_state_dict(best_model_wts)
        mlflow.log_image(fig2data(result_plot(history)), "result_plot.png")
        mlflow.log_param("Estimated Threshold", threshold)
        mlflow.pyfunc.log_model(artifact_path=mlflow_experiment_name, python_model=LSTMLungStudyWrapper(model=best_model, threshold=threshold), conda_env=lstm_conda_env, registered_model_name=mlflow_experiment_name, code_path=["model-builder/ml_models/anomaly_detection/lstm.py"])

if __name__ == "__main__":
    main()
