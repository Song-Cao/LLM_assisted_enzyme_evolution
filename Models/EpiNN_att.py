import torch
import torch.nn as nn
import torch.nn.functional as F

class EpiNN_att(nn.Module):


    def __init__(self, L, D, D_hidden=32, n_heads=4, dropout_rate=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.D_hidden = D_hidden or D
        assert D_hidden % n_heads == 0, "D_hidden must be divisible by n_heads"
        self.head_dim = D_hidden // n_heads

        # First-order additive components
        self.token_linear = nn.Linear(D, 1, bias=False)
        self.seq_linear   = nn.Linear(L, 1, bias=True)

        # Multi-Head Low-Rank Projection
        self.projection = nn.Linear(D, D_hidden, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.interaction_scale = nn.Parameter(torch.tensor(1.0))


    def forward(self, x, esm_priors=None, only_first_order=False):
        N, L, _ = x.shape

        # --- First-order fitness effect calculation ---
        first_order = self.token_linear(x).squeeze(-1)       # (N, L)
        first_order = self.seq_linear(first_order).squeeze(-1)  # (N,)

        # --- CHANGED: Early return for Phase 1 ---
        if only_first_order:
            return first_order

        # --- second-order fitness effect calculation ---
        h = self.projection(self.dropout(x))
        h = h.view(N, L, self.n_heads, self.head_dim).transpose(1, 2) # (N, H, L, Dh)
        attn_scores = torch.matmul(h, h.transpose(-1, -2)) / math.sqrt(self.head_dim) # (N, H, L, L)
        attn_scores = attn_scores.mean(dim=1) # Average over heads: (N, L, L)

        if esm_priors is not None:
            attn_scores = attn_scores * esm_priors

        second_order = torch.triu(attn_scores, diagonal=1).sum(dim=(1, 2)) * self.interaction_scale

        return first_order + second_order
