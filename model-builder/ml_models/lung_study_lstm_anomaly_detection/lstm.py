import torch
from torch import nn
import mlflow

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
        self.input_dimensionality: int = input_dimensionality
        self.input_dim: int = input_dim  # It is 1d, remember
        self.latent_dim: int = latent_dim
        self.num_layers: int = num_layers
        self.encoder = torch.nn.LSTM(self.input_dim, self.latent_dim, self.num_layers)
        # You can have any latent dim you want, just output has to be exact same size as input
        # In this case, only encoder and decoder, it has to be input_dim though
        self.decoder = torch.nn.LSTM(self.latent_dim, self.input_dim, self.num_layers)

    def forward(self, input):
        # Save original size first:
        original_shape = input.shape
        # Flatten 2d (or 3d or however many you specified in constructor)
        input = input.reshape(input.sh  ape[: -self.input_dimensionality] + (-1,))

        # Rest goes as in my previous answer
        _, (last_hidden, _) = self.encoder(input)
        encoded = last_hidden.repeat(input.shape)
        y, _ = self.decoder(encoded)

        # You have to reshape output to what the original was
        reshaped_y = y.reshape(original_shape)
        return torch.squeeze(reshaped_y)

class MnistTorchRNN(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = mlflow.pytorch.load_model(
            context.artifacts["torch-rnn-model"], map_location="cpu")
        self.model.to('cpu')
        self.model.eval()

    def predict(self, context, input_df):
        import numpy as np
        with torch.no_grad():
            input_tensor = torch.from_numpy(
                input_df.values.reshape(-1, 28, 28).astype(np.float32)).to('cpu')
            model_results = self.model(input_tensor).numpy()
            return np.power(np.e, model_results)