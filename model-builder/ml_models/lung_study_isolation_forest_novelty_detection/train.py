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
sys.path.append(os.path.abspath('model-builder/'))
from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
from dataloader.querybuilder import QueryBuilder
sys.path.append(os.path.abspath('.'))
from model_class.lung_study import  LungStudy

import mlflow
import mlflow.sklearn

import mlflow.pyfunc

def set_env_vars():
    os.environ["MLFLOW_URL"] = "http://127.0.0.1:5000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = ""
    os.environ["AWS_SECRET_ACCESS_KEY"] = ""

class IsolationForestWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, threshold):
          self.model = model
          self.threshold = threshold

    def predict(self, context, model_input):
        raw_data, raw_data_index = model_input[0], model_input[1]
        raw_data = raw_data.reshape(raw_data.shape[0], -1)
        return self.model.predict(raw_data).tolist()


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
    lung_study = LungStudy()
    dbconn = PostgresPandasWrapper(dbname="features", user="radar", password="bulgaria:STICK:cause",host="127.0.0.1", port=5434)
    dbconn.connect()
    #print(dbconn.get_response(cols=["*"], dataset="wine_dataset"))
    query = lung_study.get_query_for_training()
    data = dbconn.get_response(query)
    dbconn.disconnect()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    #Set experiment
    mlflow.set_experiment("lung_study_isolation_forest_novelty_detection")

    n_estimator = args.n_estimator
    train_x, train_x_index = lung_study.preprocess_data(data)
    train_x = train_x.reshape(train_x.shape[0], -1)
    with mlflow.start_run():
        est = IsolationForest(n_estimators=n_estimator)
        est.fit(train_x)
        #Returns -1 for outliers and 1 for inliers.
        detected_anamoly = est.predict(train_x)
        # Printing info about all the detected anamolys
        print("Detected Anomalies: ")
        for i in np.where(detected_anamoly == -1)[0]:
            print(train_x_index[i])

        threshold = min(est.score_samples(train_x))

        mlflow.log_param("n_estimator", n_estimator)
        mlflow.log_param("Estimated Threshold", threshold)
        mlflow.log_metric("total_anamolies_detected", np.sum(detected_anamoly == -1))

        mlflow.pyfunc.log_model(artifact_path="lung_study_isolation_forest_novelty_detection", python_model=IsolationForestWrapper(model=est, threshold=threshold), conda_env=isolation_forest_conda_env)
