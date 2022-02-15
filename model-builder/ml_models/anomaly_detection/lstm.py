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

class LSTMAnomalyDataset(Dataset):
    def __init__(self, data, data_index):
        self.data = torch.FloatTensor(data)
        self.data_index = data_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]