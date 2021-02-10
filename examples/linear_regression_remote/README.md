# MLFlow Linear Regression Remote

It saves both metadata and artifacts on mlflow-server docker stack.

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

## Running Linear Regression code

To run it, use:

```bash
 mlflow run examples/linear_regression_remote
```

You can change hyperparameters using:

```bash
python examples/linear_regression_remote/train.py --alpha 0.2 --l1-ratio 0.8
```

## Results

You can observe the results on `http://127.0.0.1:5000` and look at the stored models on `http://127.0.0.1:9000/`
