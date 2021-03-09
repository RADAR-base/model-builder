import mlflow
from mlflow.tracking import MlflowClient
from fastapi import HTTPException
from dotenv import load_dotenv
import os

class MlflowInterface():

    def __init__(self):
        self.load_env_file()
        self.mlflow_tracking_uri, self.mlflow_registry_uri = self.get_mlflow_uris()
        self.client = MlflowClient(tracking_uri=self.mlflow_tracking_uri, registry_uri=self.mlflow_registry_uri)

    def load_env_file(self):
        load_dotenv('.env')

    def get_mlflow_uris(self):
        return os.environ.get('MLFLOW_TRACKING_URI'), os.environ.get('MLFLOW_S3_ENDPOINT_URL')

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
        print(all_runs[0])
        if len(all_runs) < version:
            raise HTTPException(404, f"Version {version} of experiment {name} does not exist")
        else:
            return all_runs[version - 1]