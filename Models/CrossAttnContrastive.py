import torch
from torch import nn
import torch.nn.functional as F


class CrossAttnContrastive(nn.Module):
    def __init__(
        self,
        d_esm: int,
        d_emb: int,
        hidden_dims: list[int],
        contrast_type: str = "subtract",
        dropout_rate: float = 0.1,
    ):
        """
        Cross‐attention contrastive model.

        Args:
        -----
        d_esm: int
            Dimension of the input ESM embeddings (hidden size).
        d_emb: int
            Dimension of the key/value embeddings and query.
        hidden_dims: list[int]
            Hidden layer sizes for the final MLP.
        contrast_type: {"subtract","concat"}
            How to form the contrastive vector.
        dropout_rate: float
            Dropout probability in the final MLP.
        """
        super().__init__()
        assert contrast_type in ("subtract", "concat")
        self.contrast_type = contrast_type
        self.d_emb = d_emb

        # Cross‐attention parameters
        self.w_k = nn.Linear(d_esm, d_emb, bias=False)
        self.w_v = nn.Linear(d_esm, d_emb, bias=False)
        self.query = nn.Parameter(torch.randn(d_emb))

        # Final MLP:
        contrast_dim = d_emb if contrast_type == "subtract" else 2 * d_emb
        mlp_dims = [contrast_dim+1] + hidden_dims + [1]
        layers = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i+1]))
            if i < len(mlp_dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
        self.output_mlp = nn.Sequential(*layers)

    def forward(self, emb1, emb2, fit1) -> torch.Tensor:
        """
        emb1, emb2: (N, L, d_esm)
        fit1: (N, 1) fitness of the first variants
        Returns:
        --------
        logits: (N,) scalar predictions
        """
        N, L, _ = emb1.shape

        # -- Cross‐attention pooling for emb1 and emb2 --
        # Compute keys and values: (N, L, d_emb)
        K1 = self.w_k(emb1)
        V1 = self.w_v(emb1)
        K2 = self.w_k(emb2)
        V2 = self.w_v(emb2)

        # Prepare query: (1, d_emb) -> (N, 1, d_emb)
        Q = self.query.unsqueeze(0).unsqueeze(1).expand(N, 1, self.d_emb)

        # Scaled dot‐product attention scores: (N, 1, L)
        scale = math.sqrt(self.d_emb)
        attn1 = F.softmax((Q @ K1.transpose(1, 2)) / scale, dim=-1)
        attn2 = F.softmax((Q @ K2.transpose(1, 2)) / scale, dim=-1)

        # Weighted sum of values: (N, 1, d_emb) -> (N, d_emb)
        v1 = (attn1 @ V1).squeeze(1)
        v2 = (attn2 @ V2).squeeze(1)

        # -- Contrastive embedding --
        if self.contrast_type == "subtract":
            c = v2 - v1          # (N, d_emb)
        else:  # "concat"
            c = torch.cat([v1, v2], dim=1)  # (N, 2*d_emb)
        c = torch.cat([c, fit1], dim=1)

        # -- Final MLP to scalar logit --
        out = self.output_mlp(c).squeeze(1)  # (N,)
        return out
