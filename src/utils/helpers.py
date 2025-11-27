"""Helper utilities for PyTorch models and training."""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> str:
    """Get the best available device (CUDA/MPS/CPU).
    
    Args:
        device: Optional specific device to use
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if device is not None:
        return device
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
    """
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)
    print(model)
    print("=" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 70)


def save_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    **kwargs,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to save the checkpoint
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        **kwargs: Additional items to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, path)
    print(f"✓ Model saved to {path}")


def load_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to the checkpoint
        optimizer: Optional optimizer to load state
        device: Device to load the model on
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"✓ Model loaded from {path}")
    
    return checkpoint


def freeze_layers(model: nn.Module, freeze_until: Optional[str] = None) -> None:
    """Freeze model layers up to a specific layer.
    
    Args:
        model: PyTorch model
        freeze_until: Name of the layer to freeze up to (None = freeze all)
    """
    freeze = True
    
    for name, param in model.named_parameters():
        if freeze_until and freeze_until in name:
            freeze = False
        
        param.requires_grad = not freeze
    
    trainable = count_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"Frozen layers: {total - trainable:,} / {total:,} parameters")
    print(f"Trainable parameters: {trainable:,}")


if __name__ == "__main__":
    print("Testing helper utilities...")
    
    # Test seed setting
    set_seed(42)
    print("✓ Random seed set to 42")
    
    # Test device detection
    device = get_device()
    print(f"✓ Best available device: {device}")
    
    # Test parameter counting with a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    n_params = count_parameters(model)
    print(f"✓ Parameter counting: {n_params:,} parameters")
    
    print_model_summary(model)
