"""Models package."""

from .eegconformer import EEGConformer, PatchEmbedding, ClassificationHead
from .eegnet import EEGNet

__all__ = [
    "EEGConformer",
    "PatchEmbedding",
    "ClassificationHead",
    "EEGNet",
]
