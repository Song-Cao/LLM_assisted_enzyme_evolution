import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class CNN_Contrastive(nn.Module):
    def __init__(self, d_esm: int, use_first_fitness: bool = False):
        """
        CNN‐based contrastive model using a 1D-ResNet18 backbone.

        Args:
        -----
        d_esm: int
            Dimension of the input ESM embeddings.
        use_first_fitness: bool
            If True, append the fitness value of the first variant before the final FC.
        """
        super().__init__()
        self.use_first_fitness = use_first_fitness
        self.inplanes = 64

        # --- initial conv/bn/relu/pool (1D) ---
        self.conv1   = nn.Conv1d(d_esm, 64,
                                 kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1     = nn.BatchNorm1d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3,
                                    stride=2, padding=1)

        # --- the 4 ResNet layers (2 blocks each) ---
        self.layer1 = self._make_layer(BasicBlock1D,  64, 2)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)

        # --- global pooling + final FC ---
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        fc_in = 512 * BasicBlock1D.expansion + (1 if use_first_fitness else 0)
        self.fc = nn.Linear(fc_in, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            stride=stride,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, emb1, emb2, fit1= None):
        """
        Args:
        -----
        emb1, emb2: (N, L, d_esm)
            ESM embeddings for sequence 1 and 2.
        fit1: (N, 1), optional
            The 1‐D fitness feature of the first variant to append before the final FC.

        Returns:
        --------
        out: (N,)
            The scalar logit/regression output.
        """
        # compute difference and permute to (N, d_esm, L)
        x = emb2 - emb1
        x = x.permute(0, 2, 1)

        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # global pooling
        x = self.avgpool(x)     # (N, 512, 1)
        x = torch.flatten(x, 1) # (N, 512)

        # 5) optionally append Fitness 1
        if self.use_first_fitness and fit1 is not None:
            x = torch.cat([x, fit1], dim=1)  # (N, 513)

        # 6) final FC -> scalar
        out = self.fc(x)  # (N, 1)
        return out.squeeze(1)  # (N,)
