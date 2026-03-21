import torch.nn as nn
from torch.nn import Module
from torchvision import models


class ResNet18(Module):
    """
    ResNet-18 with frozen backbone and a tunable MLP classification head.

    Args:
        num_classes:  Number of output classes.
        hidden_dim:   Intermediate size of the MLP head.
        dropout:      Dropout probability in the MLP head.
        pretrained:   Load ImageNet weights when True.
    """
    def __init__(self, num_classes: int, hidden_dim: int = 256,
                 dropout: float = 0.4, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features  # 512
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ResNet50(Module):
    """
    ResNet-50 with frozen backbone and a tunable MLP classification head.

    Args:
        num_classes:  Number of output classes.
        hidden_dim:   Intermediate size of the MLP head.
        dropout:      Dropout probability in the MLP head.
        pretrained:   Load ImageNet weights when True.
    """
    def __init__(self, num_classes: int, hidden_dim: int = 512,
                 dropout: float = 0.4, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features  # 2048
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)
