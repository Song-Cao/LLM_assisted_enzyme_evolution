import torch
from torch import nn
import torch.nn.functional as F


class TransformerBlock1D(nn.Module):
    def __init__(self, d_model, nhead=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        # separate Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2  = nn.LayerNorm(d_model)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # x: (N, L, d_model)
        # Pre-normalization
        y = self.norm1(x)
        # Project to Q, K, V and transpose for MHA
        q = self.q_proj(y).transpose(0,1)  # (L, N, d_model)
        k = self.k_proj(y).transpose(0,1)
        v = self.v_proj(y).transpose(0,1)
        # Scaled dot-product
        y2, _ = self.attn(q, k, v)
        # back to (N, L, d_model)
        x = x + y2.transpose(0,1)
        # Feed-forward with residual
        y = self.norm2(x)
        x = x + self.ff(y)
        return x

class CT_Contrastive(nn.Module):
    def __init__(
        self,
        d_esm: int,
        d_model: int = 512,
        nhead: int = 8,
        dim_ff: int = 2048,
        num_groups: int = 3,
        blocks_per_group: int = 2,
        max_len: int = 101,
        use_first_fitness: bool = False,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.use_first_fitness = use_first_fitness
        # Stem projection + positional embedding
        self.proj    = nn.Linear(d_esm, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Hierarchical transformer groups
        self.groups = nn.ModuleList()
        for _ in range(num_groups):
            blocks = [
                TransformerBlock1D(d_model, nhead, dim_ff, dropout_rate)
                for _ in range(blocks_per_group)
            ]
            self.groups.append(nn.Sequential(*blocks))

        # Global pooling + final FC
        self.pool = nn.AdaptiveAvgPool1d(1)
        fc_in = d_model + (1 if use_first_fitness else 0)
        self.fc  = nn.Linear(fc_in, 1)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, fit1: torch.Tensor = None):
        # emb1, emb2: (N, L, d_esm)
        x = emb2 - emb1  # (N, L, d_esm)

        # Stem + position
        x = self.proj(x)  # (N, L, d_model)
        N, L, _ = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        # Transformer groups + pooling
        for group in self.groups:
            x = group(x)  # (N, L, d_model)
            x = F.avg_pool1d(
                x.transpose(1,2), kernel_size=2, stride=2
            ).transpose(1,2)  # halve length

        # Global pool
        x = self.pool(x.transpose(1,2)).squeeze(-1)  # (N, d_model)

        # Optional fitness concat
        if self.use_first_fitness and fit1 is not None:
            x = torch.cat([x, fit1], dim=1)

        # Final regression
        return self.fc(x).squeeze(-1)
