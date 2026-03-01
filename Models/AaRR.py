mport torch
import torch.nn as nn
import torch.nn.functional as F

class AaRR(nn.Module):
    def __init__(self, L, D, dropout_rate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.token_linear = nn.Linear(D, 1, bias=True)
        self.res_weight = nn.Parameter(torch.ones(1, L) * 0.1, requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, esm_priors=None, only_first_order=False):
        x = self.dropout(x)
        mask = (x.abs().sum(dim=-1) > 0).float()
        w = self.token_linear(x).squeeze(-1) + self.res_weight  # (N, L)
        w = w * mask
        w = w.unsqueeze(-1)          # (N, L, 1)
        x = (x * w).sum(dim=1)    # (N, D)  weighted sum
        x = self.mlp(x).squeeze(-1)
        return x
