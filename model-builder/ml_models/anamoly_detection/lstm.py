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
        self.embedding_dim = 2* latent_dim
        self.num_layers: int = num_layers
        self.seq_len = 5
        self.encoded_vec_dims = self.seq_len * self.embedding_dim
        self.encoder_1 = torch.nn.LSTM(self.input_feature_len, self.latent_dim, self.num_layers, batch_first=True)
        self.relu_enc_1 = nn.ReLU()
        self.encoder_2 = torch.nn.LSTM(self.latent_dim, self.embedding_dim, self.num_layers, batch_first=True)
        self.relu_enc_2 = nn.ReLU()
        self.decoder_1 = torch.nn.LSTM(self.encoded_vec_dims, self.embedding_dim, self.num_layers, batch_first=True)
        self.relu_dec_1 = nn.ReLU()
        self.decoder_2 = torch.nn.LSTM(self.embedding_dim, self.latent_dim, self.num_layers, batch_first=True)
        self.relu_dec_2 = nn.ReLU()
        self.output_layer = nn.Linear(self.latent_dim * self.seq_len, self.input_feature_len)

        nn.init.xavier_uniform_(self.encoder_1.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.encoder_2.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.decoder_1.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.decoder_2.weight_ih_l0, gain=np.sqrt(2))

    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        x, _ = self.encoder_1(x)
        x = self.relu_enc_1(x)
        x, (hidden_n, _) = self.encoder_2(x)
        encoded_vec = hidden_n.transpose(0,1).reshape((batch_size, -1))
        encoded_vec = self.relu_enc_2(encoded_vec)
        x = encoded_vec.repeat(1, self.seq_len).view(batch_size, self.seq_len, self.encoded_vec_dims)
        x, _ = self.decoder_1(x)
        x = self.relu_dec_1(x)
        x, _ = self.decoder_2(x)
        x = self.relu_dec_2(x)
        x = x.reshape(batch_size, -1)
        x = self.output_layer(x)
        return x

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
        criterion = nn.L1Loss(reduction='mean', reduce=False).to(device)
        predictions  = []
        for seq_true in dataloader:
            y_true = seq_true[:,-1,:]
            seq_true = seq_true.to(device)
            seq_pred = self.model(seq_true)
            loss = criterion(seq_pred, y_true)
            loss = torch.mean(loss, 1)
            predictions+= (loss < self.threshold).tolist()
        return predictions


    def predict(self, context, model_input):
        raw_data, raw_data_index = model_input[0], model_input[1]
        prediction_dataset = LSTMAnomalyDataset(raw_data, raw_data_index)
        prediction_dataloader = DataLoader(prediction_dataset, batch_size=4)
        return self._predict(prediction_dataloader)


class LSTMAnomalyDataset(Dataset):
    def __init__(self, data, data_index):
        self.data = torch.FloatTensor(data)
        self.data_index = data_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]