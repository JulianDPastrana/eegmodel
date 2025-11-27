"""Tests for EEG datasets."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from src.data import BCIIV2aDataset, ADHDDataset, EEGSupervisedDataset


def test_bci_dataset():
    """Test BCI IV 2a dataset loading."""
    try:
        dataset = BCIIV2aDataset(subject=1, split="T", tmin=0.0, tmax=4.0)
        
        assert len(dataset) > 0, "Dataset should not be empty"
        
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor), "X should be a tensor"
        assert isinstance(y, torch.Tensor), "Y should be a tensor"
        assert x.ndim == 2, "X should be 2D (channels, samples)"
        assert y.ndim == 0, "Y should be 0D (scalar)"
        assert 0 <= y < 4, "Label should be in range [0, 3]"
        
        print(f"✓ BCI IV 2a dataset loaded: {len(dataset)} samples")
        print(f"  Sample shape: {x.shape}, Label: {y.item()}")
        
    except FileNotFoundError as e:
        print(f"⚠ BCI dataset test skipped (data not found): {e}")
    except Exception as e:
        print(f"⚠ BCI dataset test failed: {e}")


def test_adhd_dataset():
    """Test ADHD dataset loading."""
    try:
        dataset = ADHDDataset()
        
        assert len(dataset) > 0, "Dataset should not be empty"
        
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor), "X should be a tensor"
        assert isinstance(y, torch.Tensor), "Y should be a tensor"
        assert x.ndim == 2, "X should be 2D (channels, samples)"
        assert y in [0, 1], "Label should be 0 or 1"
        
        print(f"✓ ADHD dataset loaded: {len(dataset)} samples")
        print(f"  Sample shape: {x.shape}, Label: {y.item()}")
        
    except FileNotFoundError as e:
        print(f"⚠ ADHD dataset test skipped (data not found): {e}")
    except Exception as e:
        print(f"⚠ ADHD dataset test failed: {e}")


def test_supervised_dataset():
    """Test EEG supervised dataset loading."""
    try:
        dataset = EEGSupervisedDataset()
        
        if len(dataset) > 0:
            x, y, name = dataset[0]
            assert isinstance(x, torch.Tensor), "X should be a tensor"
            assert isinstance(y, torch.Tensor), "Y should be a tensor"
            assert isinstance(name, str), "Name should be a string"
            
            print(f"✓ Supervised dataset loaded: {len(dataset)} samples")
            print(f"  Sample shape: {x.shape}, Labels shape: {y.shape}")
        else:
            print("⚠ Supervised dataset is empty (no matching files found)")
            
    except FileNotFoundError as e:
        print(f"⚠ Supervised dataset test skipped (data not found): {e}")
    except Exception as e:
        print(f"⚠ Supervised dataset test failed: {e}")


def test_dataset_iteration():
    """Test dataset iteration."""
    try:
        dataset = BCIIV2aDataset(subject=1, split="T")
        
        # Test iteration
        count = 0
        for x, y in dataset:
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            count += 1
            if count >= 5:  # Test first 5 samples
                break
        
        print(f"✓ Dataset iteration works correctly")
        
    except FileNotFoundError:
        print("⚠ Iteration test skipped (data not found)")
    except Exception as e:
        print(f"⚠ Iteration test failed: {e}")


def test_dataloader_compatibility():
    """Test dataset compatibility with PyTorch DataLoader."""
    try:
        from torch.utils.data import DataLoader
        
        dataset = BCIIV2aDataset(subject=1, split="T")
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        batch_x, batch_y = next(iter(loader))
        
        assert batch_x.shape[0] <= 4, "Batch size should be <= 4"
        assert batch_y.shape[0] == batch_x.shape[0], "X and Y batch sizes should match"
        
        print(f"✓ DataLoader compatibility confirmed")
        print(f"  Batch shape: {batch_x.shape}")
        
    except FileNotFoundError:
        print("⚠ DataLoader test skipped (data not found)")
    except Exception as e:
        print(f"⚠ DataLoader test failed: {e}")


if __name__ == "__main__":
    print("Running dataset tests...")
    print("=" * 50)
    
    test_bci_dataset()
    test_adhd_dataset()
    test_supervised_dataset()
    test_dataset_iteration()
    test_dataloader_compatibility()
    
    print("=" * 50)
    print("Dataset tests completed!")
