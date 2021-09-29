import mlflow
from mlflow.tracking import MlflowClient
from fastapi import HTTPException
from dotenv import load_dotenv
import os
import json
import pandas as pd
import sys
sys.path.insert(1, '../model-builder')
sys.path.insert(1, '..')
from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
from model_class import ModelClass
import importlib
class MlflowInterface():

    def __init__(self):
        self.load_env_file()
        self.mlflow_tracking_uri, self.mlflow_registry_uri = self.get_mlflow_uris()
        self.client = MlflowClient(tracking_uri=self.mlflow_tracking_uri, registry_uri=self.mlflow_registry_uri)
        self._get_postgres_data()

    def load_env_file(self):
        load_dotenv('.env')

    def get_mlflow_uris(self):
        return os.environ.get('MLFLOW_TRACKING_URI'), os.environ.get('MLFLOW_S3_ENDPOINT_URL')

    def _get_postgres_data(self):
        self.postgres_data = {}
        self.postgres_inference_data = {}
        self.postgres_data["user"] = os.environ.get('POSTGRES_USER')
        self.postgres_data["password"] = os.environ.get('POSTGRES_PASS')
        self.postgres_data["host"] = os.environ.get('POSTGRES_HOST')
        self.postgres_data["port"] = os.environ.get('POSTGRES_PORT')
        self.postgres_inference_data["user"] = os.environ.get('INFERENCE_POSTGRES_USER')
        self.postgres_inference_data["password"] = os.environ.get('INFERENCE_POSTGRES_PASS')
        self.postgres_inference_data["host"] = os.environ.get('INFERENCE_POSTGRES_HOST')
        self.postgres_inference_data["port"] = os.environ.get('INFERENCE_POSTGRES_PORT')

    def _search_experiment_by_name(self, name):
        experiment = self.client.get_experiment_by_name(name)
        if experiment is None:
            raise HTTPException(404, f"Experiment by name {name} does not exist")
        return experiment

    def _get_all_experiment_runs(self, experiment_id):
        return self.client.search_runs(experiment_id)

    def get_all_experiments(self):
        return self.client.list_experiments()

    def get_model_info(self, name):
        experiment = self._search_experiment_by_name(name)
        return self._get_all_experiment_runs(experiment.experiment_id)

    def get_model_version_info(self, name, version):
        experiment = self._search_experiment_by_name(name)
        all_runs = self._get_all_experiment_runs(experiment.experiment_id)
        if len(all_runs) < version:
            raise HTTPException(404, f"Version {version} of experiment {name} does not exist")
        else:
            return all_runs[len(all_runs) - version]

    def _mlflow_inference(self, model_run, df):
        if df is None:
            raise HTTPException(404, f"No data available for inference")
        loaded_model = mlflow.pyfunc.load_model(model_run.info.artifact_uri + "/" + json.loads(model_run.data.tags["mlflow.log-model.history"])[0]["artifact_path"])
        return list(loaded_model.predict(df))

    def _convert_data_to_df(self, data):
        data_format = data.format
        if data_format == "pandas_split":
            df = pd.DataFrame(columns=data.columns, data = data.data)
        elif data_format == "pandas_record":
            df = pd.DataFrame.from_records(data.record)
        elif data_format == "tf-instances":
            df = pd.DataFrame(data.instances)
        elif data_format == "tf-inputs":
            df = pd.DataFrame(data.inputs)
        return df


    def _get_data_from_postgres(self, metadata):
        postgres = PostgresPandasWrapper(dbname=metadata.dbname, **self.postgres_data)
        postgres.connect()
        module = importlib.import_module(f"model_class.{metadata.filename}")
        data_class = getattr(module, metadata.classname)
        self.data_class_instance = data_class()
        if not isinstance(self.data_class_instance, ModelClass):
            raise HTTPException(400, f"Requested class is not an instance of ModelClass")
        queries = self.data_class_instance.get_query_for_prediction(metadata.user_id, metadata.starttime, metadata.endtime)
        raw_data = postgres.get_response(queries)
        return self.data_class_instance.preprocess_data(raw_data)

    def _insert_inference_data_in_postgres(self, metadata, inference_data):
        postgres = PostgresPandasWrapper(dbname=metadata.dbname, **self.postgres_inference_data)
        postgres.connect()
        module = importlib.import_module(f"model_class.{metadata.filename}")
        data_class = getattr(module, metadata.classname)
        data_class_instance = data_class()
        if not isinstance(data_class_instance, ModelClass):
            raise HTTPException(400, f"Requested class is not an instance of ModelClass")
        postgres.insert_data(inference_data, data_class_instance.inference_table_name)


    def get_inference(self, name, version, data):
        experiment_run = self.get_model_version_info(name, version)
        df = self._convert_data_to_df(data)
        return self._mlflow_inference(experiment_run, df)

    def get_inference_with_metadata(self, name, version, metadata):
        experiment_run = self.get_model_version_info(name, version)
        df = self._get_data_from_postgres(metadata)
        inference = self._mlflow_inference(experiment_run, df)
        return_obj = self.data_class_instance.create_return_obj(df[1], name, version, inference)
        self._insert_inference_data_in_postgres(metadata, return_obj)
        return return_obj.to_dict(orient='records')

    def _get_best_model(self, name):
        experiment = self._search_experiment_by_name(name)
        print( self.client.search_runs(experiment.experiment_id, order_by=["metrics.m DESC"]))
        best_model = self.client.search_runs(experiment.experiment_id, order_by=["metrics.m DESC"])[0]
        return best_model, "best"

    def get_inference_from_best_model(self, name, data):
        experiment_run = self._get_best_model(name)
        df = self._convert_data_to_df(data)
        return self._mlflow_inference(experiment_run, df)

    def get_inference_from_best_model_with_metadata(self, name, metadata):
        experiment_run, version = self._get_best_model(name)
        df = self._get_data_from_postgres(metadata)
        return_obj = {}
        return_obj["model_name"] = name
        return_obj["version"] = version
        return_obj["inference"] = self._mlflow_inference(experiment_run, df)
        return return_obj

    def _get_latest_model(self, name):
        experiment = self._search_experiment_by_name(name)
        all_experiments = self._get_all_experiment_runs(experiment.experiment_id)
        latest_model = all_experiments[0]
        version = len(all_experiments)
        return latest_model, version

    def get_inference_from_latest_model(self, name, data):
        experiment_run = self._get_latest_model(name)
        df = self._convert_data_to_df(data)
        return self._mlflow_inference(experiment_run, df)

    def get_inference_from_latest_model_with_metadata(self, name, metadata):
        experiment_run, version = self._get_latest_model(name)
        df = self._get_data_from_postgres(metadata)
        return_obj = {}
        return_obj["model_name"] = name
        return_obj["version"] = version
        return_obj["inference"] = self._mlflow_inference(experiment_run, df)
        return return_obj