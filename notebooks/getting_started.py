"""Example notebook for getting started with EEG model training.

This notebook demonstrates:
1. Loading datasets
2. Creating models
3. Training and evaluation
4. Visualization

Run this in Jupyter or as a Python script.
"""

# %% Imports
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from src.models import EEGConformer
from src.data import BCIIV2aDataset
from src.training import Trainer
from src.utils import set_seed, get_device, print_model_summary, plot_training_history

# %% Set random seed for reproducibility
set_seed(42)

# %% Get device
device = get_device()
print(f"Using device: {device}")

# %% Load dataset
print("\nLoading BCI IV 2a dataset...")
try:
    dataset = BCIIV2aDataset(subject=1, split="T", tmin=0.0, tmax=4.0)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"  Channels: {len(dataset.ch_names)}")
    print(f"  Sampling frequency: {dataset.sfreq} Hz")
    
    # Get sample
    x, y = dataset[0]
    print(f"  Sample shape: {x.shape}, Label: {y.item()}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("Make sure the BCI IV 2a data is in data/raw/BCI_IV_2a/")
    sys.exit(1)

# %% Split dataset
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_set, val_set, test_set = random_split(
    dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
)

print(f"\nDataset split:")
print(f"  Train: {n_train} samples")
print(f"  Val: {n_val} samples")
print(f"  Test: {n_test} samples")

# %% Create data loaders
batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# %% Create model
print("\nCreating EEGConformer model...")
model = EEGConformer(
    emb_size=64,
    num_channels=len(dataset.ch_names),
    num_classes=4,
    num_heads=8,
    num_layers=3,
    dropout=0.4,
)

print_model_summary(model)

# %% Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=criterion,
    device=device,
    checkpoint_dir="checkpoints",
)

# %% Train model
print("\nTraining model...")
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,  # Use more epochs for better results
    verbose=True,
)

# %% Plot training history
plot_training_history(history)

# %% Evaluate on test set
from src.training import validate_epoch

print("\nEvaluating on test set...")
test_metrics = validate_epoch(test_loader, model, criterion, device=device, verbose=True)

print("\nTest Results:")
print(f"  Loss: {test_metrics['loss']:.4f}")
print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall: {test_metrics['recall']:.4f}")
print(f"  F1-Score: {test_metrics['f1']:.4f}")

# %% Visualize some predictions
model.eval()
with torch.no_grad():
    # Get a batch from test set
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch.to(device)
    
    # Make predictions
    outputs = model(X_batch)
    predictions = outputs.argmax(dim=1).cpu()

# Plot confusion matrix
from src.utils import plot_confusion_matrix
from src.training.metrics import compute_confusion_matrix

# Get all predictions
all_preds = []
all_targets = []
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        outputs = model(X)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(y)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

cm = compute_confusion_matrix(all_targets, all_preds)
plot_confusion_matrix(cm, class_names=["Left", "Right", "Feet", "Tongue"])

print("\n✓ Training complete!")
