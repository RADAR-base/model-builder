## Ml Models

### Lung study Isolation Forest

Compile docker file in the parent `model-builder` directory

Build docker file using the following command:

```bash
docker build -t lung_study_isolation_forest_novelty_detection -f model-builder/ml_models/Dockerfile  .
```

Run the docker file using the following command:

```bash
docker run --env conda_file_path=model-builder/ml_models/lung_study_isolation_forest_novelty_detection/conda.yaml --env conda_env_name=lung_study_isolation_forest_novelty_detection  --env ml_training_file=model-builder/ml_models/lung_study_isolation_forest_novelty_detection/train.py  lung_study_isolation_forest_novelty_detection
```

### Lstm Anamoly Detection

Compile docker file in the parent `model-builder` directory

Build docker file using the following command:

```bash
docker build -t lung_study_lstm_anomaly_detection -f model-builder/ml_models/Dockerfile  .
```

Run the docker file using the following command:

```bash
docker run --env conda_file_path=model-builder/ml_models/anamoly_detection/lung_study/conda.yaml --env conda_env_name=lung_study_lstm_anomaly_detection  --env ml_training_file=model-builder/ml_models/anamoly_detection/lung_study/train.py  lung_study_lstm_anomaly_detection
```