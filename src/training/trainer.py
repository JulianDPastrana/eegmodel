"""Training utilities and Trainer class."""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics


def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """Train the model for one epoch.
    
    Args:
        dataloader: Training data loader
        model: PyTorch model
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    iterator = tqdm(dataloader, desc="Training") if verbose else dataloader
    
    for batch_idx, (X, y) in enumerate(iterator):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.append(pred.argmax(1).cpu())
        all_targets.append(y.cpu())
        
        if verbose and batch_idx % 100 == 0:
            current = (batch_idx + 1) * len(X)
            iterator.set_postfix({"loss": f"{loss.item():.4f}", "samples": f"{current}/{len(dataloader.dataset)}"})
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics


def validate_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """Validate the model for one epoch.
    
    Args:
        dataloader: Validation data loader
        model: PyTorch model
        loss_fn: Loss function
        device: Device to validate on ('cpu' or 'cuda')
        verbose: Whether to print progress
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    iterator = tqdm(dataloader, desc="Validation") if verbose else dataloader
    
    with torch.no_grad():
        for X, y in iterator:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            
            total_loss += loss.item()
            all_preds.append(pred.argmax(1).cpu())
            all_targets.append(y.cpu())
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss
    
    return metrics


class Trainer:
    """Trainer class for managing the training loop.
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            verbose: Whether to print progress
        
        Returns:
            Training history dictionary
        """
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-" * 50)
            
            # Training
            train_metrics = train_epoch(
                train_loader,
                self.model,
                self.loss_fn,
                self.optimizer,
                self.device,
                verbose,
            )
            
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            
            if verbose:
                print(
                    f"Train - Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.4f}"
                )
            
            # Validation
            if val_loader is not None:
                val_metrics = validate_epoch(
                    val_loader,
                    self.model,
                    self.loss_fn,
                    self.device,
                    verbose,
                )
                
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])
                
                if verbose:
                    print(
                        f"Val   - Loss: {val_metrics['loss']:.4f}, "
                        f"Acc: {val_metrics['accuracy']:.4f}"
                    )
                
                # Save best model
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint("best_loss.pt", epoch, val_metrics)
                
                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.save_checkpoint("best_acc.pt", epoch, val_metrics)
        
        return self.history
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            epoch: Current epoch number
            metrics: Optional metrics to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
        
        Returns:
            Checkpoint dictionary
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        
        return checkpoint


# Legacy functions for backward compatibility
def train_loop(dataloader, model, loss_fn, optimizer):
    """Legacy training loop function (deprecated)."""
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    """Legacy test loop function (deprecated)."""
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
