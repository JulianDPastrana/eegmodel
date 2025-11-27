"""Preprocessing utilities for EEG signals."""

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch


def apply_bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to the signal.
    
    Args:
        signal: Input signal array
        lowcut: Low frequency cutoff in Hz
        highcut: High frequency cutoff in Hz
        fs: Sampling frequency in Hz
        order: Filter order (default: 5)
    
    Returns:
        Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def apply_notch_filter(
    signal: np.ndarray,
    freq: float,
    fs: float,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply a notch filter to remove a specific frequency (e.g., 50/60 Hz powerline noise).
    
    Args:
        signal: Input signal array
        freq: Frequency to remove in Hz
        fs: Sampling frequency in Hz
        quality_factor: Quality factor (default: 30.0)
    
    Returns:
        Filtered signal
    """
    b, a = iirnotch(freq, quality_factor, fs)
    return filtfilt(b, a, signal, axis=-1)


def data_windowing(
    data: np.ndarray,
    fs: int,
    window_duration: int = 2,
    overlap: float = 0.5,
) -> np.ndarray:
    """Create overlapping windows from continuous EEG data.
    
    Args:
        data: Input EEG data of shape (channels, samples)
        fs: Sampling frequency in Hz
        window_duration: Duration of each window in seconds (default: 2)
        overlap: Overlap ratio between consecutive windows (default: 0.5)
    
    Returns:
        Windowed data of shape (num_windows, channels, window_samples)
    """
    window_size = window_duration * fs
    stride = int(window_size * (1 - overlap))
    
    subject_data = np.lib.stride_tricks.sliding_window_view(
        x=data, window_shape=window_size, axis=-1
    )
    
    # Extract windows with the specified stride
    windowed = subject_data[:, ::stride, :]
    
    # Rearrange to (num_windows, channels, window_samples)
    return np.moveaxis(windowed, 1, 0)


def find_signal_in_mat(path: str) -> np.ndarray:
    """Find and extract the main signal array from a .mat file.
    
    Args:
        path: Path to the .mat file
    
    Returns:
        Signal array from the .mat file
    
    Raises:
        ValueError: If no suitable signal array is found
    """
    mat_dict = loadmat(path, simplify_cells=True)
    
    # Look for non-metadata numpy arrays
    for key, value in mat_dict.items():
        if not key.startswith("__") and isinstance(value, np.ndarray):
            return value
    
    raise ValueError(f"No suitable signal array found in {path}")


def normalize_signal(signal: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize EEG signal.
    
    Args:
        signal: Input signal of shape (channels, samples) or (trials, channels, samples)
        method: Normalization method - 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized signal
    """
    if method == "zscore":
        mean = np.mean(signal, axis=-1, keepdims=True)
        std = np.std(signal, axis=-1, keepdims=True)
        return (signal - mean) / (std + 1e-8)
    
    elif method == "minmax":
        min_val = np.min(signal, axis=-1, keepdims=True)
        max_val = np.max(signal, axis=-1, keepdims=True)
        return (signal - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "robust":
        median = np.median(signal, axis=-1, keepdims=True)
        iqr = np.percentile(signal, 75, axis=-1, keepdims=True) - \
              np.percentile(signal, 25, axis=-1, keepdims=True)
        return (signal - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing utilities...")
    
    # Create dummy signal
    fs = 250  # Hz
    duration = 10  # seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Signal with multiple frequencies
    signal = np.sin(2 * np.pi * 10 * t) + \
             0.5 * np.sin(2 * np.pi * 50 * t) + \
             0.3 * np.random.randn(len(t))
    
    signal = signal.reshape(1, -1)  # Shape: (1 channel, samples)
    
    # Test bandpass filter
    filtered = apply_bandpass_filter(signal, lowcut=8, highcut=30, fs=fs)
    print(f"✓ Bandpass filter: {signal.shape} -> {filtered.shape}")
    
    # Test notch filter
    notched = apply_notch_filter(signal, freq=50, fs=fs)
    print(f"✓ Notch filter: {signal.shape} -> {notched.shape}")
    
    # Test windowing
    windowed = data_windowing(signal, fs=fs, window_duration=2, overlap=0.5)
    print(f"✓ Windowing: {signal.shape} -> {windowed.shape}")
    
    # Test normalization
    normalized = normalize_signal(signal, method="zscore")
    print(f"✓ Z-score normalization: mean={normalized.mean():.4f}, std={normalized.std():.4f}")
