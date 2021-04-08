import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
from mlflow.tracking import MlflowClient
# Get the current working directory
sys.path.append(os.path.abspath('.'))

from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
from dataloader.querybuilder import QueryBuilder

import mlflow
import mlflow.sklearn

import mlflow.pyfunc

def set_env_vars():
    os.environ["MLFLOW_URL"] = "http://172.16.1.21:5000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://172.16.1.21:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://172.16.1.21:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = ""
    os.environ["AWS_SECRET_ACCESS_KEY"] = ""

class IsolationForestWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
          self.model = model

    def predict(self, context, model_input):
        self.model.predict(model_input)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    set_env_vars()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimator", default=100),
    args = parser.parse_args()

    isolation_forest_conda_env={'channels': ['defaults'],
     'name':'isolation_forest_conda_env',
     'dependencies': [ 'python=3.8', 'pip',
     {'pip':['mlflow','scikit-learn','cloudpickle','pandas','numpy']}]}

    # Importing query builder
    querybuilder = QueryBuilder(tablename="")
    # Read the wine-quality data from the Postgres database using dataloader
    # ADD DATABASE DETAILS HERE
    dbconn = PostgresPandasWrapper(dbname="", user="", password="")
    dbconn.connect()
    #print(dbconn.get_response(cols=["*"], dataset="wine_dataset"))
    query = querybuilder.get_all_columns()
    data = dbconn.get_response(query)
    dbconn.disconnect()

    mlflow.set_tracking_uri("http://172.16.1.21:5000")
    #Set experiment
    mlflow.set_experiment("lung_study_isolation_forest_novelty_detection")

    #print(f"tracking_uri: {mlflow.get_tracking_uri()}")
    #print(f"artifact_uri: {mlflow.get_artifact_uri()}")

    n_estimator = args.n_estimator
    train_x = data
    with mlflow.start_run():
        est = IsolationForest(n_estimators=n_estimator)
        est.fit(train_x)
        anomaly_score = est.score(train_x)

        mlflow.log_param("n_estimator", n_estimator)
        mlflow.log_metric("anomaly_score", anomaly_score)

        mlflow.pyfunc.log_model(artifact_path="lung_study_isolation_forest_novelty_detection", python_model=IsolationForestWrapper(model=est,), conda_env=isolation_forest_conda_env)
