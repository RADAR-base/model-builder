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
from dotenv import load_dotenv
# Get the current working directory
sys.path.append(os.path.abspath('model-builder/'))

from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
from dataloader.querybuilder import QueryBuilder
sys.path.append(os.path.abspath('.'))
from model_class.wine_study import WineStudy

import mlflow
import mlflow.sklearn

class ElasticNetWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
          self.model = model

    def predict(self, context, model_input):
        raw_data, raw_data_index = model_input[0], model_input[1]
        raw_data = raw_data.drop(["quality"], axis=1)
        return self.model.predict(raw_data).tolist()

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", default=0.1),
    parser.add_argument("--l1-ratio", default=0.2)
    parser.add_argument("--num-iterations", default=1000)
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
    wine_study = WineStudy()
    postgres_data = get_postgres_data()
    # Read the wine-quality data from the Postgres database using dataloader
    # ADD DATABASE DETAILS HERE
    dbconn =  PostgresPandasWrapper(dbname=postgres_data["dbname"], user=postgres_data["user"], password=postgres_data["password"],host=postgres_data["host"], port=postgres_data["port"])
    dbconn.connect()
    queries = wine_study.get_query_for_training()
    data = dbconn.get_response(queries)
    dbconn.disconnect()
    return wine_study.preprocess_data(data)

def split_data(data):
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    return train_x, train_y, test_x, test_y

def main():
    np.random.seed(42)
    args = argparser()
    load_dotenv('model-builder/ml_models/linear_regression_wine_dataset/.env')

    linear_regression_prediction_conda={'channels': ['defaults'],
     'name':'linear_regression_prediction_conda',
     'dependencies': [ 'python=3.8', 'pip',
     {'pip':['mlflow','scikit-learn','cloudpickle','pandas','numpy']}]}

    data, data_index = import_data()
    mlflow_tracking_uri, mlflow_registry_uri, mlflow_experiment_name = get_mlflow_uris()
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    #Set experiment
    mlflow.set_experiment(mlflow_experiment_name)

    train_x, train_y, test_x, test_y = split_data(data)

    alpha = float(args.alpha)
    l1_ratio = float(args.l1_ratio)
    num_iterations = int(args.num_iterations)

    with mlflow.start_run(tags={"alias":"elasticnetws"}):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=num_iterations)
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

        mlflow.pyfunc.log_model(artifact_path=mlflow_experiment_name, python_model=ElasticNetWrapper(model=lr,), conda_env=linear_regression_prediction_conda)


if __name__ == "__main__":
    main()