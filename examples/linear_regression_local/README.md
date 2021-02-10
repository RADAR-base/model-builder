# MLFlow Linear Regression Local

It saves both metadata and artifacts locally in the model-builder directory.

To run it, use:

```bash
 mlflow run examples/linear_regression_local
```

You can change hyperparameters using,

```bash
mlflow run examples/linear_regression_local -P alpha=0.5 -P l1_ratio=0.8
```
