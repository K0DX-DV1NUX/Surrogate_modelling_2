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

        # Sequence prediction layer
        self.output_layer = nn.Conv1d(
            in_channels=128,
            out_channels=self.out_features,
            kernel_size=1
        )

    def forward(self, x):
        """
        x: (batch, seq_len, in_features)

        returns:
        (batch, seq_len, out_features)
        """

        # (B, W, F) -> (B, F, W)
        x = x.permute(0, 2, 1)

        features = self.feature_extractor(x)      # (B,128,W)

        out = self.output_layer(features)         # (B,out_features,W)

        out = out.permute(0, 2, 1)                # (B,W,out_features)

        return out