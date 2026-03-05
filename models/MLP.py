import torch
import torch.nn as nn
import torch.functional as F



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.window_size = configs.window_size
        self.input_size = configs.window_size * configs.in_features
        self.out_features = configs.out_features

        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_size, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(128, self.window_size)   # predict full sequence
            )
            for _ in range(self.out_features)
        ])

    def forward(self, x):
        """
        x: (batch, seq_len, in_features)

        returns:
        (batch, seq_len, out_features)
        """

        outputs = []

        for net in self.networks:
            out = net(x)                     # (batch, window_size)
            out = out.unsqueeze(-1)          # (batch, window_size, 1)
            outputs.append(out)

        out = torch.cat(outputs, dim=-1)     # (batch, window_size, out_features)

        return out