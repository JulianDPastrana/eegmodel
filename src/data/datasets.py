"""EEG Dataset classes for different datasets."""

import glob
import os
from typing import Callable, Optional

import mne
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .preprocessing import data_windowing, find_signal_in_mat


class BCIIV2aDataset(Dataset):
    """BCI Competition IV 2a motor imagery dataset.
    
    This dataset contains EEG recordings from 9 subjects performing 4 motor imagery tasks:
    left hand, right hand, feet, and tongue movements.
    
    Args:
        root: Path to the dataset directory containing .edf files
        subject: Subject number (1-9)
        split: 'T' for training or 'E' for evaluation
        tmin: Start time of epoch in seconds
        tmax: End time of epoch in seconds
        transform: Optional transform to apply to the data
    """
    
    _label_dict = {
        "769": ("left", 0),
        "770": ("right", 1),
        "771": ("feet", 2),
        "772": ("tongue", 3),
    }

    def __init__(
        self,
        root: str = "data/raw/BCI_IV_2a/",
        subject: int = 1,
        split: str = "T",
        tmin: float = 0.0,
        tmax: float = 4.0,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.split = split
        
        if not os.path.isabs(root):
            # Make path relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            root = os.path.join(project_root, root)
        
        eeg, label = self._load_data_for_subject(root, subject, tmin, tmax)
        self.X = eeg
        self.y = label

    def _load_data_for_subject(
        self, root: str, subj: int, tmin: float, tmax: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load EEG data for a specific subject."""
        file_path = os.path.join(root, f"A{subj:02d}{self.split}.edf")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Couldn't find {file_path}")

        raw = mne.io.read_raw_edf(
            input_fname=file_path,
            eog=["EOG-left", "EOG-central", "EOG-right"],
            preload=True,
            verbose=False,
        )
        raw.pick("eeg")

        self.sfreq = raw.info["sfreq"]
        self.ch_names = raw.info["ch_names"]

        events, ann_map = mne.events_from_annotations(raw)

        candidates = {
            "T1": ("left", 0),
            "T2": ("right", 1),
            "T3": ("feet", 2),
            "T4": ("tongue", 3),
            "769": ("left", 0),
            "770": ("right", 1),
            "771": ("feet", 2),
            "772": ("tongue", 3),
            "Stimulus/769": ("left", 0),
            "Stimulus/770": ("right", 1),
            "Stimulus/771": ("feet", 2),
            "Stimulus/772": ("tongue", 3),
        }

        event_id, label_map = {}, {}
        for desc, code in ann_map.items():
            for key, (name, lab) in candidates.items():
                if key in desc:
                    event_id[name] = code
                    label_map[code] = lab

        if not event_id:
            raise RuntimeError(
                "Couldn't find class cue annotations (769, 770, 771, 772). "
                f"File name: A{subj:02d}{self.split}.edf"
            )

        wanted_codes = list(label_map.keys())
        cue_mask = np.isin(events[:, 2], wanted_codes)
        cue_events = events[cue_mask]

        epochs = mne.Epochs(
            raw,
            cue_events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )

        eeg = torch.from_numpy(epochs.get_data()).float()
        event_codes = torch.from_numpy(epochs.events[:, 2])
        label = torch.tensor(
            [label_map[int(c)] for c in event_codes], dtype=torch.long
        )

        return eeg, label

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y


class ADHDDataset(Dataset):
    """ADHD vs Control EEG dataset.
    
    This dataset contains filtered EEG recordings from ADHD patients and control subjects.
    Data is automatically windowed into 2-second segments.
    
    Args:
        root: Path to the dataset directory
        window_duration: Duration of each window in seconds (default: 2)
        overlap: Overlap ratio between windows (default: 0.5)
    """
    
    def __init__(
        self,
        root: str = "data/raw/EEG-DATASET-ADHD-CONTROL/",
        window_duration: int = 2,
        overlap: float = 0.5,
    ):
        if not os.path.isabs(root):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            root = os.path.join(project_root, root)
        
        self.fs = 128  # Sampling frequency
        self.window_duration = window_duration
        self.overlap = overlap
        
        # Find all .mat files
        control_paths = glob.glob(os.path.join(root, "Control/**/*.mat"), recursive=True)
        adhd_paths = glob.glob(os.path.join(root, "ADHD/**/*.mat"), recursive=True)
        
        self.data = []
        self.labels = []
        
        # Load and window control data (label = 0)
        for path in control_paths:
            try:
                signal = find_signal_in_mat(path)
                windowed = data_windowing(signal, self.fs, window_duration, overlap)
                self.data.append(torch.from_numpy(windowed).float())
                self.labels.extend([0] * len(windowed))
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        # Load and window ADHD data (label = 1)
        for path in adhd_paths:
            try:
                signal = find_signal_in_mat(path)
                windowed = data_windowing(signal, self.fs, window_duration, overlap)
                self.data.append(torch.from_numpy(windowed).float())
                self.labels.extend([1] * len(windowed))
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        # Concatenate all windows
        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class EEGSupervisedDataset(Dataset):
    """Supervised EEG dataset with behavioral annotations.
    
    This dataset pairs EEG recordings (.mat files) with behavioral data from Excel files.
    It filters trials based on the 'Missing' and 'Inhibition' columns.
    
    Args:
        root_dir: Root directory containing .mat and .xlsx files
        eeg_key: Key to access EEG data in .mat files (default: "data")
    """
    
    def __init__(self, root_dir: str = "data/raw", eeg_key: str = "data"):
        if not os.path.isabs(root_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            root_dir = os.path.join(project_root, root_dir)
        
        self.eeg_key = eeg_key
        self.data = []

        mat_files = []
        excel_files = {}

        # Search recursively for all files
        for root, _, files in os.walk(root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                if f.endswith(".mat"):
                    mat_files.append(full_path)
                elif f.endswith(".xlsx") and f.startswith("yn_"):
                    base = os.path.splitext(f)[0].replace("yn_", "")
                    excel_files[base] = full_path

        # Match and process the data
        for mat_path in mat_files:
            file_name = os.path.basename(mat_path)
            if "_CORTADO_" not in file_name:
                continue

            base_name = file_name.replace("_CORTADO_", "_").replace(".mat", "")
            if base_name not in excel_files:
                continue

            excel_path = excel_files[base_name]

            # Load EEG
            mat_data = loadmat(mat_path)
            if self.eeg_key not in mat_data:
                print(f"⚠️ Key '{self.eeg_key}' not found in {mat_path}")
                continue
            eeg_data = np.array(mat_data[self.eeg_key])

            # Load Excel
            behavior = pd.read_excel(excel_path)

            if "Missing" not in behavior.columns:
                print(f"⚠️ Column 'Missing' not found in {excel_path}")
                continue

            # Filter rows where 'Missing' == 1
            behavior = behavior[behavior["Missing"] == 1]

            if len(behavior) != eeg_data.shape[0]:
                print(
                    f"❌ Skipping {base_name} due to dimension mismatch "
                    f"(EEG={eeg_data.shape[0]}, Excel={len(behavior)})"
                )
                continue

            first_col = behavior.columns[0]
            mask_5 = behavior[first_col] == 5

            # Apply the same filter to both behavior and EEG
            behavior = behavior[mask_5]
            eeg_data = eeg_data[mask_5.values, :]

            # Convert labels to 0/1
            y = behavior["Inhibition"].astype(str)
            y = np.array([1.0 if val == "1" or val == 1 else 0.0 for val in y])
            y_tensor = torch.from_numpy(y).long()

            eeg_tensor = torch.from_numpy(eeg_data).float()

            # Store the data
            self.data.append((eeg_tensor, y_tensor, base_name))

        if not self.data:
            raise ValueError(
                "No valid pairs of .mat and .xlsx files found."
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.data[idx]


if __name__ == "__main__":
    # Test BCI IV 2a dataset
    print("=" * 50)
    print("Testing BCI Competition IV 2a dataset...")
    print("=" * 50)
    
    try:
        ds = BCIIV2aDataset(subject=1, split="E", tmin=0.0, tmax=4.0)
        print(f"✓ Loaded {len(ds)} epochs")
        print(f"✓ Sampling frequency: {ds.sfreq} Hz")
        print(f"✓ Channels ({len(ds.ch_names)}): {ds.ch_names[:3]}...")
        
        x, y = ds[0]
        print(f"✓ Sample shape: x={x.shape}, y={y}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Testing ADHD dataset...")
    print("=" * 50)
    
    try:
        ds_adhd = ADHDDataset()
        print(f"✓ Loaded {len(ds_adhd)} windows")
        x, y = ds_adhd[0]
        print(f"✓ Sample shape: x={x.shape}, y={y}")
    except Exception as e:
        print(f"✗ Error: {e}")
