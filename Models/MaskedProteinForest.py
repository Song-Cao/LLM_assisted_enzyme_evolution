import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedProteinForest(nn.Module):
    def __init__(self, L, D, hidden_dim=64, k=16, temp=1.0): # Increased temp
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1) * temp) # Make it learnable

        self.feature_gate = nn.Linear(D, hidden_dim)
        self.feature_map  = nn.Linear(D, hidden_dim)
        self.residue_scorer = nn.Linear(hidden_dim, 1)

        # Add a small amount of dropout to the bottleneck
        self.pos_bottleneck = nn.Linear(L, k)
        self.dropout = nn.Dropout(0.1)
        self.pos_to_output  = nn.Linear(k, 1)

    def forward(self, x, esm_priors=None, only_first_order=False):
        mask = (x.abs().sum(dim=-1) > 0).float()

        # Prevent division by zero if temp is learned
        t = torch.clamp(self.temp, min=0.1)
        gate = torch.sigmoid(self.feature_gate(x) / t)

        val = self.feature_map(x)
        h = val * gate

        res_scores = self.residue_scorer(h).squeeze(-1)
        masked_scores = res_scores * mask

        # USE LEAKY_RELU to prevent constant output collapse
        latent_pos = F.leaky_relu(self.pos_bottleneck(masked_scores), negative_slope=0.01)
        out = self.pos_to_output(self.dropout(latent_pos)).squeeze(-1)

        return out
