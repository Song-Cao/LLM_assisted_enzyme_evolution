import torch
import torch.nn as nn
import torch.nn.functional as F

class EpiNN_aaindex(nn.Module):
    def __init__(self, L, D, dropout_rate=0.0):
        super().__init__()
        self.L = L
        self.D = D
        self.dropout = nn.Dropout(dropout_rate)
        self.token_linear = nn.Linear(L*D+1, 1)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(2*D, D),
            nn.LeakyReLU(),
            nn.Linear(D, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )
        self.interaction_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, emb, attn=None, only_first_order=False):
        # compute first order interactions
        x1 = self.dropout(emb)
        x1 = self.token_linear(x1).squeeze(-1)

        # compute second order interactions
        x2 = emb[:,:-1].reshape(emb.shape[0],self.L, self.D)
        mask = (x2.abs().sum(dim=-1) > 0).float()  # (N, L)
        mask = mask[:, :, None] * mask[:, None, :] # (N, L, L)
        w = self.token_linear.weight.squeeze(0)[:-1].view(1, self.L, self.D)
        x2 = x2 * w
        xi = x2.unsqueeze(2).expand(-1, -1, self.L, -1)  # (N, L, L, D) copies x[:, i, :]
        xj = x2.unsqueeze(1).expand(-1, self.L, -1, -1)  # (N, L, L, D) copies x[:, j, :]
        x2 = torch.cat([(xi+xj)/2, (xi-xj).abs()], dim=-1)          # (N, L, L, 2D)
        x2 = self.interaction_mlp(x2).squeeze(-1) * mask   # (N, L, L)
        x2 = torch.triu(x2, diagonal=1).sum(dim=(1, 2)) * self.interaction_scale
        return x1 + x2
