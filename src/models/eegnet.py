import torch.nn as nn
from torch import Tensor


class EEGNet(nn.Module):
    r"""
    EEGNet: Compact Convolutional Neural Network for EEG classification

    Tensor Flow
    ---------------
    Input : Tensor[(B, C, T)]
    Output: Tensor[(B, num_classes)]

    Parameters
    --------------
    B : Batch size
    C : Number of electrodes (num_electrodes)
    T : Samples per chunk (chunk_size)
    """

    def __init__(
        self,
        num_electrodes: int = 32,
        kernel_1: int = 64,
        kernel_2: int = 16,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        pool1: int = 4,
        pool2: int = 8,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # Block 1: Temporal -> Spatial
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernel_1),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=F1),
            nn.Conv2d(
                in_channels=F1,
                out_channels=D * F1,
                kernel_size=(num_electrodes, 1),
                padding="valid",
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=D * F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1)),
            nn.Dropout2d(p=dropout),
        )
        # Block 2: Separable (Depthwise -> Pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=D * F1,
                kernel_size=(1, kernel_2),
                padding="same",
                groups=D * F1,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=D * F1,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2)),
            nn.Dropout2d(p=dropout),
            nn.Flatten(),
        )
        # Classifier
        self.classifier = nn.LazyLinear(
            out_features=num_classes,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (B, C, T)
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.block1(x)  # (B, D * F1, 1, T // pool1)
        x = self.block2(x)  # (B, F2 * (T // (pool1 * pool2)))
        return self.classifier(x)  # (B, num_classes)
