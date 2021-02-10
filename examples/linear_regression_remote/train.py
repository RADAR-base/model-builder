import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from mlflow.tracking import MlflowClient
# Get the current working directory
sys.path.append(os.path.abspath('.'))

from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper

import mlflow
import mlflow.sklearn

def set_env_vars():
    os.environ["MLFLOW_URL"] = "http://127.0.0.1:5000"
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://127.0.0.1:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "AKIAIOSFODNN7EXAMPLE"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    set_env_vars()
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha")
    parser.add_argument("--l1-ratio")
    args = parser.parse_args()

    # Read the wine-quality data from the Postgres database using dataloader
    dbconn = PostgresPandasWrapper(dbname="opendb", user="heet", password="password")
    dbconn.connect()
    #print(dbconn.get_response(cols=["*"], dataset="wine_dataset"))
    data = dbconn.get_response(cols=["*"], tablename="wine_dataset")
    dbconn.disconnect()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
     # #Set experiment
    mlflow.set_experiment("test-experiment")
    #print(f"tracking_uri: {mlflow.get_tracking_uri()}")
    #print(f"artifact_uri: {mlflow.get_artifact_uri()}")
    # Create an experiment with a name that is unique and case sensitive.
    #client = MlflowClient()
    #experiment_id = client.set_experiment("test1")

    # Fetch experiment metadata information
    #experiment = client.get_experiment(2)
    #print("Name: {}".format(experiment.name))
    #print("Experiment_id: {}".format(experiment.experiment_id))
    #print("Artifact Location: {}".format(experiment.artifact_location))
    #print("Tags: {}".format(experiment.tags))
    #print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(args.alpha)
    l1_ratio = float(args.l1_ratio)

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "linear_regression_model")
