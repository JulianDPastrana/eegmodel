"""Train EEGConformer model on BCI IV 2a or ADHD dataset."""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import EEGConformer
from src.data import BCIIV2aDataset, ADHDDataset
from src.training import Trainer
from src.utils import set_seed, get_device, print_model_summary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train EEGConformer model")
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="bci",
        choices=["bci", "adhd"],
        help="Dataset to use (bci or adhd)",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=1,
        help="Subject number for BCI IV 2a dataset (1-9)",
    )
    
    # Model arguments
    parser.add_argument("--emb-size", type=int, default=64, help="Embedding size")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    
    return parser.parse_args()


def load_dataset(args):
    """Load and split the dataset."""
    if args.dataset == "bci":
        print(f"Loading BCI IV 2a dataset (Subject {args.subject})...")
        dataset = BCIIV2aDataset(subject=args.subject, split="T")
        num_channels = len(dataset.ch_names)
        num_classes = 4  # left, right, feet, tongue
    elif args.dataset == "adhd":
        print("Loading ADHD dataset...")
        dataset = ADHDDataset()
        # Get channel info from first sample
        sample_x, _ = dataset[0]
        num_channels = sample_x.shape[0]
        num_classes = 2  # ADHD vs Control
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    n_test = n_total - n_train - n_val
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset split - Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader, num_channels, num_classes


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    train_loader, val_loader, test_loader, num_channels, num_classes = load_dataset(args)
    
    # Create model
    print("\nCreating EEGConformer model...")
    model = EEGConformer(
        emb_size=args.emb_size,
        num_channels=num_channels,
        num_heads=args.num_heads,
        dim_feedfoward=args.dim_feedforward,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    )
    
    print_model_summary(model)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    
    # Handle class imbalance if needed
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Train model
    print(f"\nTraining for {args.epochs} epochs...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        verbose=True,
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    from src.training import validate_epoch
    
    test_metrics = validate_epoch(
        test_loader, model, criterion, device=device, verbose=True
    )
    
    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1']:.4f}")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    trainer.save_checkpoint("final_model.pt", args.epochs, test_metrics)
    print(f"\nFinal model saved to {final_path}")
    
    # Plot training history
    from src.utils import plot_training_history
    
    plot_path = os.path.join(args.checkpoint_dir, "training_history.png")
    plot_training_history(history, save_path=plot_path)
    print(f"Training history plot saved to {plot_path}")


if __name__ == "__main__":
    main()
