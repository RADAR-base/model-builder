import torch
from torch import nn
import mlflow
import mlflow.pyfunc
import numpy as np
from torch.utils.data import Dataset, DataLoader

class LSTM(nn.Module):
    # input_dim has to be size after flattening
    # For 20x20 single input it would be 400
    def __init__(
        self,
        input_dimensionality: int,
        input_dim: int,
        latent_dim: int,
        num_layers: int,
    ):
        super(LSTM, self).__init__()
        self.input_feature_len = input_dimensionality
        self.input_dim: int = input_dim
        self.latent_dim: int = latent_dim
        self.num_layers: int = num_layers
        self.encoder = torch.nn.LSTM(self.input_feature_len, self.latent_dim, self.num_layers)
        self.decoder = torch.nn.LSTM(self.latent_dim, self.input_feature_len, self.num_layers)

        nn.init.xavier_uniform(self.encoder.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform(self.decoder.weight_ih_l0, gain=np.sqrt(2))

    def forward(self, input):
        input = input.float()
        # Flatten 2d (or 3d or however many you specified in constructor)
        #input = input.reshape(input.shape[: -self.input_dimensionality] + (-1,))

        # Rest goes as in my previous answer
        encoded, _ = self.encoder(input)
        y, _ = self.decoder(encoded)

        return y

class LSTMWrapper(mlflow.pyfunc.PythonModel):

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
        criterion = nn.L1Loss(reduction='sum').to(device)
        predictions  = []
        for seq_true in dataloader:
            seq_true = seq_true.to(device)
            seq_pred = self.model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append((loss < self.threshold).tolist())
        return predictions


    def predict(self, context, model_input):
        raw_data, raw_data_index = model_input[0], model_input[1]
        prediction_dataset = LungStudyDataset(raw_data, raw_data_index)
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=8)
        return self._predict(prediction_dataloader).tolist()

    def predict(self, context, input_df):
        import numpy as np
        with torch.no_grad():
            input_tensor = torch.from_numpy(
                input_df.values.reshape(-1, 28, 28).astype(np.float32)).to('cpu')
            model_results = self.model(input_tensor).numpy()
            return np.power(np.e, model_results)

class LungStudyDataset(Dataset):
    def __init__(self, data, data_index):
        self.data = data
        self.data_index = data_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]