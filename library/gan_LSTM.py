import numpy as np
import torch
from torch import nn

from constants import WINDOW_SIZE, N_ASSETS
from tcn import TemporalBlock

class Generator(nn.Module):
    """
    Generator: LSTM architecture
    """

    # Define noise size
    NOISE_WINDOW_SIZE = WINDOW_SIZE
    NOISE_SIZE = NOISE_WINDOW_SIZE * 10

    # Define LSTM parameters
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=self.NOISE_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True
        )
        self.linear = nn.Linear(self.HIDDEN_SIZE, WINDOW_SIZE * N_ASSETS)

    def forward(self, x):
        # Reshape input for LSTM (batch_size, sequence_length=1, input_size)
        x = x.unsqueeze(1)
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        # Take the last output and pass through linear layer
        output = self.linear(lstm_out[:, -1, :])
        return output

    @classmethod
    def get_noise(cls, batch_size: int) -> torch.tensor:
        """
        (batch_size, noise_size)
        """
        return torch.randn(batch_size, cls.NOISE_SIZE)

    @classmethod
    def get_shifted_noise(cls, batch_size: int) -> torch.tensor:
        """
        (batch_size, noise_size, window_size)
        Each observation in batch contains last values from previous batch and new value
        """
        if batch_size == 1:
            return cls.get_noise(batch_size)

        # Generate base noise
        noise = torch.randn(cls.NOISE_SIZE, cls.NOISE_WINDOW_SIZE + batch_size - 1)
        result = torch.zeros(batch_size, cls.NOISE_SIZE * cls.NOISE_WINDOW_SIZE)
        for i in range(batch_size):
            result[i] = torch.concatenate([noise[j, i:i + cls.NOISE_WINDOW_SIZE] for j in range(N_ASSETS)])
        return result

class Discriminator(nn.Module):
    """
    Discriminator: 1 to 1 Causal temporal convolutional network with skip connections.
    This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """
    HIDDEN_CHANNELS = 10

    def __init__(self, seq_len=WINDOW_SIZE):
        super().__init__()
        self.tcn = nn.ModuleList([TemporalBlock(N_ASSETS, self.HIDDEN_CHANNELS, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(self.HIDDEN_CHANNELS, self.HIDDEN_CHANNELS, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8]]])
        self.last = nn.Conv1d(self.HIDDEN_CHANNELS, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()