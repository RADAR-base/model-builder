import mlflow
from mlflow.tracking import MlflowClient
from fastapi import HTTPException
from dotenv import load_dotenv
import os
import json
import pandas as pd
import sys
sys.path.insert(1, '../model-builder')
from dataloader.postgres_pandas_wrapper import PostgresPandasWrapper
import importlib
class MlflowInterface():

    def __init__(self):
        self.load_env_file()
        self.mlflow_tracking_uri, self.mlflow_registry_uri = self.get_mlflow_uris()
        self.client = MlflowClient(tracking_uri=self.mlflow_tracking_uri, registry_uri=self.mlflow_registry_uri)
        self._get_postgres_data()
        self.postgresql= PostgresPandasWrapper(self.postgres_data)

    def load_env_file(self):
        load_dotenv('.env')

    def get_mlflow_uris(self):
        return os.environ.get('MLFLOW_TRACKING_URI'), os.environ.get('MLFLOW_S3_ENDPOINT_URL')

    def _get_postgres_data(self):
        self.postgres_data = {}
        self.postgres_data["user"] = os.environ.get('POSTGRES_USER')
        self.postgres_data["password"] = os.environ.get('POSTGRES_PASS')
        self.postgres_data["host"] = os.environ.get('POSTGRES_HOST')
        self.postgres_data["port"] = os.environ.get('POSTGRES_PORT')

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

    def _make_query_from_metadata(self, metadata):
        if metadata.columns is None:
            return f"SELECT * FROM {metadata.tablename} WHERE time between {metadata.starttime} and {metadata.endtime}"
        return ""

    def _get_data_from_postgres(self, metadata):
        self.postgres.connect()
        module = importlib.import_module(f"..model_class.{metadata.classname}")
        data_class = getattr(module, metadata.classname)
        data_class_instance = data_class(metadata.user_id, metadata.participant_id)
        query = data_class_instance.get_query()
        return self.postgres.get_response(query)


    def get_inference(self, name, version, data):
        experiment_run = self.get_model_version_info(name, version)
        df = self._convert_data_to_df(data)
        return self._mlflow_inference(experiment_run, df)

    def get_inference_with_metadata(self, name, version, metadata):
        experiment_run = self.get_model_version_info(name, version)
        df = self._get_data_from_postgres(metadata)
        return self._mlflow_inference(experiment_run, df)

    def _get_best_model(self, name):
        experiment = self._search_experiment_by_name(name)
        best_model = self.client.search_runs(experiment.experiment_id, order_by=["metrics.m DESC"])[0]
        return best_model

    def get_inference_from_best_model(self, name, data):
        experiment_run = self._get_best_model(name)
        df = self._convert_data_to_df(data)
        return self._mlflow_inference(experiment_run, df)

    def get_inference_from_best_model_with_metadata(self, name, metadata):
        experiment_run = self._get_best_model(name)
        df = self._get_data_from_postgres(metadata)
        return self._mlflow_inference(experiment_run, df)

    def _get_latest_model(self, name):
        experiment = self._search_experiment_by_name(name)
        latest_model = self._get_all_experiment_runs(experiment.experiment_id)[0]
        return latest_model

    def get_inference_from_latest_model(self, name, data):
        experiment_run = self._get_latest_model(name)
        df = self._convert_data_to_df(data)
        return self._mlflow_inference(experiment_run, df)

    def get_inference_from_latest_model_with_metadata(self, name, metadata):
        experiment_run = self._get_latest_model(name)
        df = self._get_data_from_postgres(metadata)
        return self._mlflow_inference(experiment_run, df)