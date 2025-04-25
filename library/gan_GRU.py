import torch
from torch import nn

from library.constants import WINDOW_SIZE, N_ASSETS
from library.tcn import TemporalBlock

###################################################################################
# Noise function
###################################################################################


class Generator(nn.Module):
    """
    Generator: 3 to 1 Causal temporal convolutional network with skip connections.
    This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """

    # Define noise size
    NOISE_WINDOW_SIZE = WINDOW_SIZE
    NOISE_SIZE = N_ASSETS * WINDOW_SIZE

    # Define number of hidden channels
    HIDDEN_DIM = 100
    NUM_LAYERS = 6
    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size

        self.gru = nn.GRU(
            input_size=self.NOISE_SIZE,
            hidden_size=self.HIDDEN_DIM,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            dropout=0.2 if self.NUM_LAYERS > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, WINDOW_SIZE * N_ASSETS),
        )

        # self.tcn = nn.ModuleList([TemporalBlock(self.NOISE_SIZE, self.HIDDEN_CHANNELS, kernel_size=1, stride=1, dilation=1, padding=0),
        #                          *[TemporalBlock(self.HIDDEN_CHANNELS, self.HIDDEN_CHANNELS, kernel_size=self.kernel_size, stride=1, dilation=i, padding=(self.kernel_size - 1) * i // 2 if self.kernel_size % 2 != 0 else i) for i in [1, 2, 4, 8]]])
        # self.last = nn.Conv1d(self.HIDDEN_CHANNELS, N_ASSETS, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        orig_noise = x  # Сохраняем исходный шум
        batch_size = x.shape[0]
        x, _ = self.gru(x)
        x = self.fc(x)
        return x.view(batch_size, N_ASSETS, WINDOW_SIZE) + 0.1 * orig_noise.view(-1, N_ASSETS, WINDOW_SIZE)

    @classmethod
    def get_noise(cls, batch_size: int) -> torch.tensor:
        """
        (batch_size, noise_size, window_size)
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
        noise = torch.randn(cls.NOISE_SIZE + batch_size - 1)
        result = torch.zeros(batch_size, cls.NOISE_SIZE)
        for i in range(batch_size):
            result[i] = torch.concatenate([noise[i:i + cls.NOISE_SIZE]])
        test = torch.randn(batch_size, cls.NOISE_SIZE)
        return test


class Discriminator(nn.Module):
    """
    Discriminator: 1 to 1 Causal temporal convolutional network with skip connections.
    This network uses 1D convolutions in order to model multiple timeseries co-dependency.
    """
    HIDDEN_CHANNELS = 10

    def __init__(self, kernel_size=2, seq_len=WINDOW_SIZE):
        super().__init__()
        self.kernel_size = kernel_size

        self.tcn = nn.ModuleList([TemporalBlock(N_ASSETS, self.HIDDEN_CHANNELS, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(self.HIDDEN_CHANNELS, self.HIDDEN_CHANNELS, kernel_size=self.kernel_size, stride=1, dilation=i, padding=(self.kernel_size - 1) * i // 2 if self.kernel_size != 2 else i) for i in [1, 2, 4, 8]]])
        self.last = nn.Conv1d(self.HIDDEN_CHANNELS, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()
