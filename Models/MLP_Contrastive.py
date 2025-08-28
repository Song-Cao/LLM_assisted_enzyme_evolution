import torch
from torch import nn
import torch.nn.functional as F


class MLP_Contrastive(nn.Module):
    def __init__(
        self,
        d_esm: int,
        embedding_hidden_dims: list[int],
        d_emb: int,
        contrast_type: str,
        output_hidden_dims: list[int],
        dropout_rate: float = 0.1,
    ):
        """
        A fully differentiable contrastive model with only MLPs.

        Args:
        -----
        d_esm: int
            Dimension of the input ESM embeddings.
        embedding_hidden_dims: list[int]
            Hidden layer sizes for the shared embedding MLP.
            e.g. [512,256] means d_esm->512->256->d_emb.
        d_emb: int
            Output dimension of the shared embedding MLP.
        contrast_type: {"subtract","concat"}
            How to form the contrastive vector.
        output_hidden_dims: list[int]
            Hidden layer sizes for the final MLP.
            e.g. [128,64] means input_dim->128->64->1.
        dropout_rate: float
        """
        super().__init__()
        assert contrast_type in ("subtract","concat")
        self.contrast_type = contrast_type

        # ---- Shared embedding MLP: d_esm -> ... -> d_emb ----
        dims = [d_esm] + embedding_hidden_dims + [d_emb]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
        self.embed_mlp = nn.Sequential(*layers)

        # ---- Final output MLP: contrastive_dim+1 -> ... -> 1 ----
        contrast_dim = d_emb if contrast_type=="subtract" else 2*d_emb
        in_dim = contrast_dim + 1
        dims = [in_dim] + output_hidden_dims + [1]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
        self.output_mlp = nn.Sequential(*layers)

    def forward(self, emb1, emb2, fit1):
        """
        emb1, emb2: (N, L, d_esm)
        fit1:       (N, 1)

        Returns:
        --------
        logits:    (N,)  scalar predictions
        """
        # 1) mean over sequence length
        v1 = emb1.mean(dim=1)  # (N, d_esm)
        v2 = emb2.mean(dim=1)

        # 2) shared embedding MLP
        e1 = self.embed_mlp(v1)  # (N, d_emb)
        e2 = self.embed_mlp(v2)

        # 3) contrastive vector
        if self.contrast_type == "subtract":
            c = e2 - e1          # (N, d_emb)
        else:  # concat
            c = torch.cat([e1, e2], dim=1)  # (N, 2*d_emb)

        # 4) append the fitness of the first variant as an 1D feature
        x = torch.cat([c, fit1], dim=1)     # (N, contrast_dim+1)

        # 5) final MLP to scalar
        out = self.output_mlp(x)           # (N, 1)
        return out.squeeze(1)              # (N,)
