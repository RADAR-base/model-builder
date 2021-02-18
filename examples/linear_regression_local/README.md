# MLFlow Linear Regression Local

It saves both metadata and artifacts locally in the model-builder directory.

Before this example, go to the README file in examples directory and add winedataset your local postgres database then add the same detail in line 37 of `train.py`.

## Creating conda Enviroment

First, install conda enviroment using:

```bash
conda env create --file examples/linear_regression_local/conda.yaml
```

Then, activate the conda enviroment using:

```bash
conda activate linear_regression_local_example
```

## Running Linear Regression code

To run it, use:

```bash
 python examples/linear_regression_local/train.py
```

You can change hyperparameters using:

```bash
python examples/linear_regression_local/train.py --alpha 0.2 --l1-ratio 0.8
```

## Results

Initiate mlflow server using the given command:

```bash
mlflow server
```

Now, You can observe the results on `http://127.0.0.1:5000`
