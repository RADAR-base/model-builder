import unittest
from unittest import mock
from fastapi.testclient import TestClient
import mlflow
import sys

def mock_inference_sum(model_run, df):
  return list(df["column_1"] + df["column_2"])

class TestAppServer(unittest.TestCase):
    def setUp(self):
        self.experiments = [
                {
                  "artifact_location": "s3://mlflow/1",
                  "experiment_id": "1",
                  "lifecycle_stage": "active",
                  "name": "experiment_1",
                  "tags": {}
                },
                {
                  "artifact_location": "s3://mlflow/2",
                  "experiment_id": "2",
                  "lifecycle_stage": "active",
                  "name": "experiment_2",
                  "tags": {}
                }
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
    def tearDown(self):
        del sys.modules['main']

    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_all_experiment_runs(self, mock_mlflow_client):
        mock_mlflow_client.return_value.list_experiments.return_value = self.experiments
        #  Importing main app after mocking few functions
        import main
        client = TestClient(main.app)
        response = client.get("/model")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.experiments)

    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_model_by_name(self, mock_mlflow_client):
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = mlflow.entities.Experiment(**self.experiments[0])
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        #  Importing main app after mocking few functions
        import main
        client = TestClient(main.app)
        response = client.get("/model/experiment_1/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.experiment_1_details)

    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_model_by_name_and_version(self, mock_mlflow_client):
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = mlflow.entities.Experiment(**self.experiments[0])
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        #  Importing main app after mocking few functions
        import main
        client = TestClient(main.app)
        response = client.get("/model/experiment_1/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.experiment_1_details[1])

    @mock.patch("mlflow_interface.MlflowInterface._mlflow_inference")
    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_inference_pandas_split(self, mock_mlflow_client, mock_mlflow_inference):
        mock_mlflow_inference.side_effect = mock_inference_sum
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = mlflow.entities.Experiment(**self.experiments[0])
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        test_input = {"columns":["column_1", "column_2"],"data":[[1,2], [0.2, 0.3], [0.45, 0.55]]}
        import main
        client = TestClient(main.app)
        response = client.post( "/model/experiment_1/1/invocation",
                        json=test_input,
                    )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [3, 0.5, 1.0])

    @mock.patch("mlflow_interface.MlflowInterface._mlflow_inference")
    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_inference_pandas_record(self, mock_mlflow_client, mock_mlflow_inference):
        mock_mlflow_inference.side_effect = mock_inference_sum
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = mlflow.entities.Experiment(**self.experiments[0])
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        test_input = {"record": [{"column_1":1, "column_2":2}, {"column_1":0.2, "column_2":0.3}, {"column_1":0.45, "column_2":0.55}]}
        import main
        client = TestClient(main.app)
        response = client.post( "/model/experiment_1/1/invocation",
                        json=test_input,
                    )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [3, 0.5, 1.0])

    @mock.patch("mlflow_interface.MlflowInterface._mlflow_inference")
    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_inference_tf_instances(self, mock_mlflow_client, mock_mlflow_inference):
        mock_mlflow_inference.side_effect = mock_inference_sum
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = mlflow.entities.Experiment(**self.experiments[0])
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        test_input = {"instances": [{"column_1":1, "column_2":2}, {"column_1":0.2, "column_2":0.3}, {"column_1":0.45, "column_2":0.55}]}
        import main
        client = TestClient(main.app)
        response = client.post( "/model/experiment_1/1/invocation",
                        json=test_input,
                    )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [3, 0.5, 1.0])

    @mock.patch("mlflow_interface.MlflowInterface._mlflow_inference")
    @mock.patch("mlflow.tracking.MlflowClient")
    def test_get_inference_tf_inputs(self, mock_mlflow_client, mock_mlflow_inference):
        mock_mlflow_inference.side_effect = mock_inference_sum
        mock_mlflow_client.return_value.get_experiment_by_name.return_value = mlflow.entities.Experiment(**self.experiments[0])
        mock_mlflow_client.return_value.search_runs.return_value = self.experiment_1_details
        test_input = {"inputs" : {"column_1": [1, 0.2, 0.45], "column_2":[2, 0.3, 0.55]}}
        import main
        client = TestClient(main.app)
        response = client.post( "/model/experiment_1/1/invocation",
                        json=test_input,
                    )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [3, 0.5, 1.0])