import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.input_size = configs.in_features
        self.hidden_size = 128
        self.num_layers = 4
        self.out_features = configs.out_features
        self.dropout = 0.2

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x):
        """
        x: (batch, seq_len, in_features)
        out: (batch, out_features) - where each column corresponds
        to a different target (e.g., SEI Rate, Temperature)
        """

        out, (_, _) = self.lstm(x) # (batch, seq_len, hidden_size)

        output = self.fc(out)       # (batch, seq_len, out_features)

        return output