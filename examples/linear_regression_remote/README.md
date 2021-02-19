# MLFlow Linear Regression Remote

It saves both metadata and artifacts on mlflow-server docker stack.

Before this example, go to the README file in examples directory and add winedataset your local postgres database then add the same detail in line 46 of `train.py`.

## Deploying mlflow-server docker stack

To deploy the  mlflow-server docker stack follow the instruction from [here](https://github.com/RADAR-base/RADAR-Docker/tree/distributed-dcompose/distributed-dcompose-stack/mlflow).

Set the enviroment variable in `examples/linear_regression_remote/conda.yaml` same as the mlflow-server.

## Creating conda Enviroment

First, install conda enviroment using:

```bash
conda env create --file examples/linear_regression_remote/conda.yaml
```

Then, activate the conda enviroment using:

```bash
conda activate linear_regression_remote_example
```

## Installing without using conda

You can install the depedencies directly too using,

```bash
pip install -r requirements.txt
pip install -r examples/linear_regression_remote/requirements.txt
```

You would need to set global enviroment variables.

```bash
export MLFLOW_URL: http://127.0.0.1:5000
export MLFLOW_TRACKING_URI: http://127.0.0.1:5000
export MLFLOW_S3_ENDPOINT_URL: http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID: <AWS KEY ID>
export AWS_SECRET_ACCESS_KEY: <AWS EXAMPLE SECRET KEY>
```


## Running Linear Regression code

To run it, use:

```bash
 python examples/linear_regression_remote/train.py
```

You can change hyperparameters using:

```bash
python examples/linear_regression_remote/train.py --alpha 0.2 --l1-ratio 0.8
```

## Results

You can observe the results on `http://127.0.0.1:5000` and look at the stored models on `http://127.0.0.1:9000/`
