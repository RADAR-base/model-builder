import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
# Get the current working directory
sys.path.append(os.path.abspath('model-builder/'))
from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
from dataloader.querybuilder import QueryBuilder
sys.path.append(os.path.abspath('.'))
from model_class.lung_study import  LungStudy

import mlflow
import mlflow.sklearn

import mlflow.pyfunc

class IsolationForestWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, threshold):
          self.model = model
          self.threshold = threshold

    def predict(self, context, model_input):
        raw_data, raw_data_index = model_input[0], model_input[1]
        raw_data = raw_data.reshape(raw_data.shape[0], -1)
        return self.model.predict(raw_data).tolist()

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimator", default=100)
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
    # Read the wine-quality data from the Postgres database using dataloader
    # ADD DATABASE DETAILS HERE
    dbconn =  PostgresPandasWrapper(dbname=postgres_data["dbname"], user=postgres_data["user"], password=postgres_data["password"],host=postgres_data["host"], port=postgres_data["port"])
    dbconn.connect()
    queries = lung_study.get_query_for_training()
    data = dbconn.get_response(queries)
    dbconn.disconnect()
    processed_data = lung_study.preprocess_data(data)
    if processed_data is None:
        raise ValueError("Not enough data for training")
    return processed_data

def main():
    np.random.seed(42)
    args = argparser()
    load_dotenv('model-builder/ml_models/lung_study_isolation_forest_novelty_detection/.env')

    isolation_forest_conda_env={'channels': ['defaults'],
     'name':'isolation_forest_conda_env',
     'dependencies': [ 'python=3.8', 'pip',
     {'pip':['mlflow','scikit-learn','cloudpickle','pandas','numpy']}]}

    mlflow_tracking_uri, mlflow_registry_uri, mlflow_experiment_name = get_mlflow_uris()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    #Set experiment
    mlflow.set_experiment(mlflow_experiment_name)

    n_estimator = args.n_estimator
    try:
        train_x, train_x_index =  import_data()
    except ValueError as e:
        with mlflow.start_run(tags={"alias":"rfad"}):
            mlflow.set_tag("LOG_STATUS", f"FAILED: {e}")
            sys.exit(1)
    train_x = train_x.reshape(train_x.shape[0], -1)
    with mlflow.start_run(tags={"alias":"rfad"}):
        est = IsolationForest(n_estimators=n_estimator)
        est.fit(train_x)
        #Returns -1 for outliers and 1 for inliers.
        detected_anamoly = est.predict(train_x)
        threshold = min(est.score_samples(train_x))
        mlfow.set_tag("LOG_STATUS", f"Successful")
        mlflow.log_param("n_estimator", n_estimator)
        mlflow.log_param("Estimated Threshold", threshold)
        mlflow.log_metric("total_anamolies_detected", np.sum(detected_anamoly == -1))
        mlflow.pyfunc.log_model(artifact_path=mlflow_experiment_name, python_model=IsolationForestWrapper(model=est, threshold=threshold), registered_model_name=mlflow_experiment_name, conda_env=isolation_forest_conda_env)

if __name__ == "__main__":
    main()