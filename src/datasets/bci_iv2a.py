import os
from typing import Callable, Optional

import mne
import torch
import numpy as np
from torch.utils.data import Dataset


class BCIIV2aDataset(Dataset):
    _label_dict = {
        "769": ("left", 0),
        "770": ("right", 1),
        "771": ("feet", 2),
        "772": ("tongue", 3),
    }

    def __init__(
        self,
        root: str = "../../data/BCI_IV_2a/",
        subject: int = 1,
        split: str = "T", # T or E
        tmin: float = 0.0,
        tmax: float = 4.0,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(script_dir, root))
        print(root)

        self.split = split

        
        eeg, label = self._load_data_for_subject(
            root, subject, tmin, tmax
        )
            
        self.X = eeg
        self.y = label

    def _load_data_for_subject(self, root: str, subj: int, tmin: float, tmax: float):

        file_path = os.path.join(root, f"A{subj:02d}{self.split}.edf")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Couldn't find {file_path}")

        raw = mne.io.read_raw_edf(
            input_fname=file_path,
            eog=["EOG-left", "EOG-central", "EOG-right"],
            preload=True
            )
        raw.pick("eeg")

        self.sfreq = raw.info["sfreq"]
        self.ch_names = raw.info["ch_names"]
    

        events, ann_map = mne.events_from_annotations(raw)

        candidates = {
        "T1": ("left", 0), "T2": ("right", 1), "T3": ("feet", 2), "T4": ("tongue", 3),
        "769": ("left", 0), "770": ("right", 1), "771": ("feet", 2), "772": ("tongue", 3),
        "Stimulus/769": ("left", 0), "Stimulus/770": ("right", 1),
        "Stimulus/771": ("feet", 2), "Stimulus/772": ("tongue", 3),
    }

        event_id, label_map = {}, {}
        for desc, code in ann_map.items():
            for key, (name, lab) in candidates.items():
                print(desc, key)
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
        )

        eeg = torch.from_numpy(epochs.get_data())
        event_codes = torch.from_numpy(epochs.events[:, 2])
        label = torch.tensor([label_map[int(c)] for c in event_codes], dtype=torch.long)

        return eeg, label

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y


if __name__ == "__main__":
    print("Loading BCI Competition IV 2a dataset...")

    ds = BCIIV2aDataset(
        subject=1,
        split="E",
        tmin=0.0,
        tmax=4.0,
    )

    print(f"Loaded {len(ds)} epochs")
    print(f"Sampling frequency: {ds.sfreq} Hz")
    print(f"Channels ({len(ds.ch_names)}): {ds.ch_names} ...")

    # rand_idx = torch.randint(0, len(ds), (1,)).item()
    for rand_idx in range(50):
        x, y = ds[rand_idx]
        print(f"Random epoch {rand_idx}: x shape = {x.shape}, y = {y}")
