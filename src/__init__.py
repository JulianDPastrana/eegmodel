"""EEG Model Training Package - PyTorch-based EEG classification models."""

__version__ = "0.1.0"

from .models import EEGConformer, EEGNet
from .data import BCIIV2aDataset, ADHDDataset, EEGSupervisedDataset
from .training import Trainer, train_epoch, validate_epoch
from .utils import set_seed, get_device, count_parameters

__all__ = [
    "EEGConformer",
    "EEGNet",
    "BCIIV2aDataset",
    "ADHDDataset",
    "EEGSupervisedDataset",
    "Trainer",
    "train_epoch",
    "validate_epoch",
    "set_seed",
    "get_device",
    "count_parameters",
]
