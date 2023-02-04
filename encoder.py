import torch
import torch.nn as nn


class Wsl_encoder(nn.Module):
    def __init__(self):
        super(Wsl_encoder, self).__init__()
        self.net = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        # Remove linear and pool layers as classification is not being performed
        self.net = nn.Sequential(*list(self.net.children())[:-2])

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x
