"""Tests for training utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.training import Trainer, train_epoch, validate_epoch
from src.training.metrics import compute_metrics, accuracy


def create_dummy_data(n_samples=100, n_features=10, n_classes=3):
    """Create dummy dataset for testing."""
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(X, y)


def create_dummy_model(n_features=10, n_classes=3):
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(n_features, 20),
        nn.ReLU(),
        nn.Linear(20, n_classes),
    )


def test_train_epoch():
    """Test single training epoch."""
    model = create_dummy_model()
    dataset = create_dummy_data()
    loader = DataLoader(dataset, batch_size=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    metrics = train_epoch(loader, model, criterion, optimizer, device="cpu", verbose=False)
    
    assert "loss" in metrics, "Metrics should contain loss"
    assert "accuracy" in metrics, "Metrics should contain accuracy"
    assert 0 <= metrics["accuracy"] <= 1, "Accuracy should be in [0, 1]"
    
    print(f"✓ Train epoch works: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")


def test_validate_epoch():
    """Test single validation epoch."""
    model = create_dummy_model()
    dataset = create_dummy_data()
    loader = DataLoader(dataset, batch_size=16)
    
    criterion = nn.CrossEntropyLoss()
    
    metrics = validate_epoch(loader, model, criterion, device="cpu", verbose=False)
    
    assert "loss" in metrics, "Metrics should contain loss"
    assert "accuracy" in metrics, "Metrics should contain accuracy"
    
    print(f"✓ Validate epoch works: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")


def test_trainer():
    """Test Trainer class."""
    import tempfile
    
    model = create_dummy_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            device="cpu",
            checkpoint_dir=tmpdir,
        )
        
        # Create train and val datasets
        train_dataset = create_dummy_data(n_samples=80)
        val_dataset = create_dummy_data(n_samples=20)
        
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Train for a few epochs
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            verbose=False,
        )
        
        assert len(history["train_loss"]) == 3, "Should have 3 epochs of train loss"
        assert len(history["val_loss"]) == 3, "Should have 3 epochs of val loss"
        
        print(f"✓ Trainer works: final train loss={history['train_loss'][-1]:.4f}")


def test_metrics():
    """Test metric computation."""
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 1, 2, 0, 2, 1])
    
    metrics = compute_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    
    acc = accuracy(y_true, y_pred)
    assert 0 <= acc <= 1, "Accuracy should be in [0, 1]"
    
    print(f"✓ Metrics computation works: accuracy={acc:.4f}")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    import tempfile
    
    model = create_dummy_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            device="cpu",
            checkpoint_dir=tmpdir,
        )
        
        # Save checkpoint
        trainer.save_checkpoint("test.pt", epoch=5, metrics={"accuracy": 0.95})
        
        # Load checkpoint
        checkpoint = trainer.load_checkpoint("test.pt")
        
        assert checkpoint["epoch"] == 5
        assert checkpoint["metrics"]["accuracy"] == 0.95
        
        print("✓ Checkpoint save/load works correctly")


if __name__ == "__main__":
    print("Running training tests...")
    print("=" * 50)
    
    test_train_epoch()
    test_validate_epoch()
    test_trainer()
    test_metrics()
    test_checkpoint_save_load()
    
    print("=" * 50)
    print("All training tests passed! ✓")
