import torch
from torch import Tensor
from eegmodels.models import EEGNet


def test_output_default_shape() -> None:
    model = EEGNet()
    x: Tensor = torch.randn(4, 32, 128)  # (B, C, T)
    out: Tensor = model(x)
    assert out.shape == (4, 2)


def test_num_classes_dim() -> None:
    model = EEGNet(num_classes=5)
    x: Tensor = torch.randn(4, 32, 128)
    out: Tensor = model(x)
    assert out.shape == (4, 5)
