import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # Добавление первого слоя
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Добавление скрытых слоев
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))

        # Добавление выходного слоя
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # Добавление функции активации ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:  # Пропускаем через все слои, кроме последнего
            x = self.relu(layer(x))
        x = self.layers[-1](x)  # Последний слой без активации
        return x