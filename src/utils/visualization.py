"""Visualization utilities for EEG data and model results."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with training history
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: Optional list of class names
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_eeg_signal(
    signal: np.ndarray,
    fs: float,
    channels: Optional[List[str]] = None,
    duration: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot EEG signals for multiple channels.
    
    Args:
        signal: EEG signal array of shape (channels, samples)
        fs: Sampling frequency in Hz
        channels: Optional list of channel names
        duration: Optional duration to plot (in seconds)
        save_path: Optional path to save the figure
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    
    n_channels, n_samples = signal.shape
    
    if duration is not None:
        n_samples = min(n_samples, int(duration * fs))
        signal = signal[:, :n_samples]
    
    time = np.arange(n_samples) / fs
    
    if channels is None:
        channels = [f"Ch {i+1}" for i in range(n_channels)]
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, n_channels * 1.5), sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    for i, (ax, ch_name) in enumerate(zip(axes, channels)):
        ax.plot(time, signal[i], linewidth=0.5)
        ax.set_ylabel(ch_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.set_title("EEG Signal", fontsize=14, fontweight="bold")
        if i == n_channels - 1:
            ax.set_xlabel("Time (s)", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_frequency_spectrum(
    signal: np.ndarray,
    fs: float,
    channel_idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """Plot frequency spectrum of an EEG signal.
    
    Args:
        signal: EEG signal array of shape (channels, samples)
        fs: Sampling frequency in Hz
        channel_idx: Index of channel to plot
        save_path: Optional path to save the figure
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    
    # Compute FFT
    n_samples = signal.shape[1]
    fft_vals = np.fft.rfft(signal[channel_idx])
    fft_freq = np.fft.rfftfreq(n_samples, 1.0 / fs)
    fft_power = np.abs(fft_vals) ** 2
    
    plt.figure(figsize=(12, 4))
    plt.plot(fft_freq, fft_power, linewidth=1)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Power", fontsize=12)
    plt.title(f"Frequency Spectrum - Channel {channel_idx}", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)  # Focus on 0-50 Hz range
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


if __name__ == "__main__":
    print("Testing visualization utilities...")
    
    # Test training history plot
    history = {
        "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2],
        "val_loss": [0.9, 0.7, 0.5, 0.4, 0.3],
        "train_acc": [0.6, 0.7, 0.8, 0.85, 0.9],
        "val_acc": [0.55, 0.65, 0.75, 0.8, 0.85],
    }
    # plot_training_history(history)  # Uncomment to visualize
    
    # Test confusion matrix
    cm = np.array([[50, 2, 0], [3, 45, 2], [1, 3, 46]])
    # plot_confusion_matrix(cm, class_names=["Class A", "Class B", "Class C"])  # Uncomment to visualize
    
    print("✓ Visualization utilities ready")
