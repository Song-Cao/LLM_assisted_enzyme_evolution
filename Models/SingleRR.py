import torch
from torch import nn
import torch.nn.functional as F


class SingleRR(nn.Module):
    def __init__(self, embedding_dim, dropout_rate=0.1):
        """
        SingleRR model for a single embedding regression (ridge regression).

        Args:
            embedding_dim (int): Dimension of the input embedding.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(SingleRR, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # Linear layer outputs a single value per sample.
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, emb):
        """
        Args:
            emb (torch.Tensor): Tensor of shape (N, embedding_dim)

        Returns:
            logits (torch.Tensor): Tensor of shape (N,), representing the predicted continuous score.
        """
        x = self.dropout(emb)
        logits = self.linear(x)  # shape: (N, 1)
        return logits.squeeze(1)  # shape: (N,)
