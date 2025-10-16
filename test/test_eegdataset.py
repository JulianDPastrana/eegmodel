import torch
from torch.utils.data import DataLoader
from src.datasets.eeg_supervised import EEGDataset

dataset = EEGDataset(root_dir=".data")
