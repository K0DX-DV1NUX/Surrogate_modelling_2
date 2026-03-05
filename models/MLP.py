import torch
import torch.nn as nn
import torch.functional as F



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_size = configs.window_size * configs.in_features # or configs.in_features depending on your data format
        self.out_features = configs.out_features
        
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(), 
                nn.Linear(self.input_size, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
            )
            for _ in range(self.out_features)
        ])
        

    def forward(self, x):
        """
        x: (batch, seq_len, in_features)
        out: (batch, out_features) - where each column corresponds
        to a different target (e.g., SEI Rate, Temperature)
        """

        outputs = []
        for i in range(self.out_features):
            out = self.networks[i](x)
            outputs.append(out)
        
        out = torch.cat(outputs, dim=-1)  # shape: (batch, out_features)
        return out