import torch.nn as nn
from torch.nn import Module
from torchvision import models


class ViT_B16(Module):
    """
    ViT-B/16 with frozen backbone, last 4 encoder blocks unfrozen,
    and a tunable MLP classification head.

    Partial unfreezing lets the transformer attend to art-style-relevant
    features while keeping earlier general-purpose representations fixed.

    Args:
        num_classes:  Number of output classes.
        hidden_dim:   Intermediate size of the MLP head.
        dropout:      Dropout probability in the MLP head.
        pretrained:   Load ImageNet weights when True.
    """
    def __init__(self, num_classes: int, hidden_dim: int = 256,
                 dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.vit_b_16(weights=weights)

        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last 4 transformer encoder blocks
        encoder_blocks = list(self.model.encoder.layers.children())
        for block in encoder_blocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze final LayerNorm
        for param in self.model.encoder.ln.parameters():
            param.requires_grad = True

        # Replace head — new layers have requires_grad=True by default
        num_ftrs = self.model.heads.head.in_features  # 768
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_ftrs, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)
