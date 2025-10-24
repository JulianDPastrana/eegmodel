import os
from typing import Callable, Optional, Union, Sequence

import torch
from torch.utils.data import Dataset
import mne


class BCIIV2aDataset(Dataset):
    _wanted = {
        "769": ("left", 0),
        "770": ("right", 1),
        "771": ("feet", 2),
        "772": ("tongue", 3),
    }

    def __init__(
        self,
        root: str = "../../data/BCI_IV_2a/",
        subjects: Union[int, Sequence[int]] = 1,
        split: str = "train",
        tmin: float = 0.0,
        tmax: float = 4.0,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(script_dir, root))
        print(root)
        if isinstance(subjects, int):
            subjects = [subjects]
        subjects = list(subjects)

        split_code = split.strip().upper()[0]
        if split_code not in ("T", "E"):
            raise ValueError("split must be one of {'train','eval','T','E'}")
        self.split = split_code

        all_X, all_y = [], []
        loaded_any = False

        for subj in subjects:
            fname = os.path.join(root, f"A{subj:02d}{self.split}.edf")
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Couldn't find {fname}")

            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
            raw.pick_types(eeg=True, eog=False, stim=False)

            events_np, ann_map = mne.events_from_annotations(raw, verbose=False)
            events = torch.from_numpy(events_np).long()

            event_id, label_map = {}, {}
            for desc, code in ann_map.items():
                for key, (name, lab) in self._wanted.items():
                    if key in desc:
                        event_id[name] = code
                        label_map[code] = lab

            if not event_id:
                raise RuntimeError(
                    "Couldn't find class cue annotations (769..772). "
                    "Ensure you're using BCI IV-2a .gdf files."
                )

            wanted_codes = torch.tensor(list(label_map.keys()), dtype=torch.long)
            cue_mask = torch.isin(events[:, 2], wanted_codes)
            cue_events = events[cue_mask].cpu().numpy()

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

            X_t = torch.from_numpy(epochs.get_data()).float()
            event_codes = torch.from_numpy(epochs.events[:, 2]).long()
            y_t = torch.empty_like(event_codes, dtype=torch.long)
            for code, lab in label_map.items():
                y_t[event_codes == code] = lab

            all_X.append(X_t)
            all_y.append(y_t)

            self.sfreq = float(raw.info["sfreq"])
            self.ch_names = list(epochs.ch_names)
            loaded_any = True

        if not loaded_any:
            raise ValueError("No subjects were loaded; check 'subjects' and data path.")

        self.X = torch.cat(all_X, dim=0)
        self.y = torch.cat(all_y, dim=0)

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
        subjects=1,
        split="train",
        tmin=0.0,
        tmax=4.0,
    )

    print(f"Loaded {len(ds)} epochs")
    print(f"Sampling frequency: {ds.sfreq} Hz")
    print(f"Channels ({len(ds.ch_names)}): {ds.ch_names} ...")
