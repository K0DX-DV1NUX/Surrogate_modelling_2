import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.in_features = configs.in_features
        self.out_features = configs.out_features

        self.width = 128
        self.modes = 16

        # lifting layer
        self.fc0 = nn.Linear(self.in_features, self.width)

        # Fourier blocks
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # projection
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, self.out_features)

    def forward(self, x):
        """
        x: (B, W, in_features)
        output: (B, W, out_features)
        """

        B, W, F = x.shape

        x = self.fc0(x)        # (B,W,width)

        x = x.permute(0, 2, 1) # (B,width,W)

        x1 = self.conv0(x)
        x = x1 + self.w0(x)
        x = torch.tanh(x)

        x1 = self.conv1(x)
        x = x1 + self.w1(x)
        x = torch.tanh(x)

        x1 = self.conv2(x)
        x = x1 + self.w2(x)
        x = torch.tanh(x)

        x1 = self.conv3(x)
        x = x1 + self.w3(x)

        x = x.permute(0, 2, 1)  # (B,W,width)

        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x
    
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)

        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (B, C, W)

        B, C, W = x.shape

        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            B, self.out_channels, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :self.modes] = torch.einsum(
            "bix, iox -> box",
            x_ft[:, :, :self.modes],
            self.weights
        )

        x = torch.fft.irfft(out_ft, n=W)

        return x