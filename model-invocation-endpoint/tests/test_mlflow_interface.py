import unittest
import pytest
from unittest import mock
import mlflow_interface
import mlflow
from mlflow.tracking import MlflowClient
from data_input_format import DataInputModel, DataInputModelSplit
import pandas as pd

def mock_inference_sum(model_run, df):
  return list(df["column_1"] + df["column_2"])

class TestMlflowInterface(unittest.TestCase):

    def setUp(self):
        self.experiments = [
                mlflow.entities.Experiment(**{
                  "artifact_location": "s3://mlflow/1",
                  "experiment_id": "1",
                  "lifecycle_stage": "active",
                  "name": "experiment_1",
                  "tags": {}
                }),
                mlflow.entities.Experiment(**{
                  "artifact_location": "s3://mlflow/2",
                  "experiment_id": "2",
                  "lifecycle_stage": "active",
                  "name": "experiment_2",
                  "tags": {}
                })
            ]
        self.experiment_1_details = [
                  {
                    "data": {
                      "metrics": {
                        "rmse": 0.80,
                        "r2": 0.15,
                        "mae": 0.62
                      },
                      "params": {
                        "alpha": "0.5",
                        "l1_ratio": "0.3"
                      },
                    },
                  },

                  {
                    "data": {
                    "metrics": {
                      "rmse": 0.7930738575377759,
                      "r2": 0.10363088394613762,
                      "mae": 0.6325736430372558
                    },
                    "params": {
                      "alpha": "0.5",
                      "l1_ratio": "0.6"
                    }
                  },
                  },
                ]

    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_all_experiment_runs(self, mock_mlflow_client):
        mock_mlflow_client.return_value.list_experiments.return_value = self.experiments
        mlflow_interface_obj = mlflow_interface.MlflowInterface()
        self.assertEqual(mlflow_interface_obj.get_all_experiments(), self.experiments)

    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_model_info(self, mock_mlflow_client):
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = self.experiments[0]
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        mlflow_interface_obj = mlflow_interface.MlflowInterface()
        self.assertEqual(mlflow_interface_obj.get_model_info("experiment_1"), self.experiment_1_details)

    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_model_version_info(self, mock_mlflow_client):
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = self.experiments[0]
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        mlflow_interface_obj = mlflow_interface.MlflowInterface()
        self.assertEqual(mlflow_interface_obj.get_model_version_info("experiment_1", 1), self.experiment_1_details[1])

    @mock.patch("mlflow_interface.MlflowInterface._mlflow_inference")
    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_inference_pandas_split(self, mock_mlflow_client, mock_mlflow_inference):
        mock_mlflow_inference.side_effect = mock_inference_sum
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = self.experiments[0]
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        test_input = {"columns":["column_1", "column_2"],"data":[[1,2], [0.2, 0.3], [0.45, 0.55]]}
        print(test_input)
        input_data = DataInputModelSplit(**test_input)
        mlflow_interface_obj = mlflow_interface.MlflowInterface()
        self.assertEqual(input_data.format, "pandas_split")
        self.assertEqual(mlflow_interface_obj.get_inference("experiment_1", 1, input_data), [3, 0.5, 1.0])