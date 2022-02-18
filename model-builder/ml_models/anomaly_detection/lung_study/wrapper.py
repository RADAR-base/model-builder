import torch
from torch import nn
import mlflow
import mlflow.pyfunc
from torch.utils.data import Dataset, DataLoader
from ..lstm_attention import LSTMAnomalyDataset


class LSTMLungStudyWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model, threshold):
          self.model = model
          self.threshold = threshold

    def _get_device(self):
        if torch.cuda.is_available():
            return "gpu"
        else:
            return "cpu"

    def _predict(self, dataloader):
        device = self._get_device()
        self.model.eval()
        criterion = nn.L1Loss(reduction='sum', reduce=False).to(device)
        predictions  = []
        predicted_vectors = []
        for seq_true in dataloader:
            y_true = seq_true[:,-1,:]
            seq_true = seq_true.to(device)
            seq_pred = self.model(seq_true)
            loss = criterion(seq_pred, y_true)
            loss = torch.sum(loss, 1)
            predicted_vectors += seq_pred.tolist()
            predictions+= (loss >= self.threshold).tolist()
        return predictions, predicted_vectors


    def predict(self, context, model_input):
        raw_data, raw_data_index = model_input[0], model_input[1]
        prediction_dataset = LSTMAnomalyDataset(raw_data, raw_data_index)
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=4)
        return self._predict(prediction_dataloader)
