## Build and Run the Dockerfile

Compile docker file in the parent `model-builder` directory

Build docker file using the following command:

```bash
docker build -t lung_study_lstm_anomaly_detection  -f model-builder/ml_models/anamoly_detection/lung_study/Dockerfile  .
```

Run the docker file using the following command:

```bash
docker run lung_study_lstm_anomaly_detection
```

### Configuring Env file

#### On Local system

If ```mlflow-server``` is running on your local system, change `127.0.0.1` to host.docker.internal for macbook or use ```--network="host"``` for linux. For more info visit [this](https://stackoverflow.com/questions/24319662/from-inside-of-a-docker-container-how-do-i-connect-to-the-localhost-of-the-mach#:~:text=Use%20%2D%2Dnetwork%3D%22host%22,for%20Linux%2C%20per%20the%20documentation.).