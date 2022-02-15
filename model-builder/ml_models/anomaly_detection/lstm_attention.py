import torch
from torch import nn
import mlflow
import mlflow.pyfunc
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(CustomAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.query_weight = nn.Parameter(torch.randn((input_dim, input_dim)), requires_grad =True)
        self.key_weight = nn.Parameter(torch.randn((input_dim, input_dim)), requires_grad =True)
        self.value_weight = nn.Parameter(torch.randn((input_dim, input_dim)), requires_grad =True)
        self.multihead_att = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)

    def forward(self, x):
        query = torch.matmul(x, self.query_weight)
        key = torch.matmul(x, self.key_weight)
        value = torch.matmul(x, self.value_weight)
        return self.multihead_att(query, key, value)



class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, enc_layers=2):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=enc_layers,
            batch_first=True
        )
        self.att1 = CustomAttentionLayer(self.hidden_dim)
        self.relu1 = nn.ReLU()

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=enc_layers,
            batch_first=True
        )
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()


    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x = self.relu1(x)
        x, _ = self.att1(x)
        x = self.relu2(x)
        x, (hidden_n, _) = self.rnn2(x)
        hidden_n = self.relu3(hidden_n)
        return hidden_n[-1]

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1, dec_layers=2):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=dec_layers,
            batch_first=True
        )
        self.att1 = CustomAttentionLayer(input_dim)
        self.relu1 = nn.ReLU()
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=dec_layers,
            batch_first=True
        )
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, 1).reshape(self.seq_len, -1, self.input_dim).swapaxes(0,1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x = self.relu1(x)
        x, _ = self.att1(x)
        x = self.relu2(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        hidden_n = self.relu3(hidden_n)
        return self.output_layer(hidden_n[-1])

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=128, enc_layers=1, dec_layers=1):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim, enc_layers)
        self.decoder = Decoder(seq_len, embedding_dim, n_features, dec_layers)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LSTMAnomalyDataset(Dataset):
    def __init__(self, data, data_index):
        self.data = torch.FloatTensor(data)
        self.data_index = data_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]