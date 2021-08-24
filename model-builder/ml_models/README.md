## Ml Models

### Lung study Isolation Forest

Compile docker file in the parent `model-builder` directory

Build docker file using the following command:

```bash
docker build -t lung_study_isolation_forest_novelty_detection --build-arg conda_file_path=model-builder/ml_models/lung_study_isolation_forest_novelty_detection/conda.yaml --build-arg conda_env_name=lung_study_isolation_forest_novelty_detection  --build-arg ml_training_file=model-builder/ml_models/lung_study_isolation_forest_novelty_detection/train.py -f model-builder/ml_models/Dockerfile  .
```

Run the docker file using the following command:

```bash
docker run lung_study_isolation_forest_novelty_detection
```

### Lstm Anamoly Detection

Compile docker file in the parent `model-builder` directory

Build docker file using the following command:

```bash
docker build -t lung_study_lstm_anomaly_detection --build-arg conda_file_path=model-builder/ml_models/anamoly_detection/lung_study/conda.yaml --build-arg conda_env_name=lung_study_lstm_anomaly_detection --build-arg ml_training_file=model-builder/ml_models/anamoly_detection/lung_study/train.py -f model-builder/ml_models/Dockerfile  .
```

Run the docker file using the following command:

```bash
docker run lung_study_lstm_anomaly_detection
```