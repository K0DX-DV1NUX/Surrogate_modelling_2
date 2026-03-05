import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.in_channels = configs.in_features
        self.out_features = configs.out_features

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.Tanh(),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.Tanh(),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Independent output heads (like your MLP networks list)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),   # Global pooling
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
            for _ in range(self.out_features)
        ])

    def forward(self, x):
        """
        x: (batch, seq_len, in_features)
        out: (batch, out_features) - where each column corresponds 
        to a different target (e.g., SEI Rate, Temperature)
        """

        # Convert to (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        features = self.feature_extractor(x)

        outputs = []
        for head in self.heads:
            out = head(features)
            outputs.append(out)

        out = torch.cat(outputs, dim=-1)

        return out