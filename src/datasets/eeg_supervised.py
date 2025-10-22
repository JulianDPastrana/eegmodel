import os
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from scipy.io import loadmat


class EEGDataset(Dataset):
    """
    Directory expectations:
      - .mat files contain EEG under key `eeg_key` (default: "data")
      - .xlsx files are named "yn_<base>.xlsx"
      - Matching rule: "<base>_CORTADO_.mat" <-> "yn_<base>.xlsx"

    Returned item for index i:
      (eeg: FloatTensor[rows_kept, T], label: LongTensor[rows_kept], base_name: str)
    """

    def __init__(self, root_dir: str, eeg_key: str = "data") -> None:
        self.root_dir: str = root_dir
        self.eeg_key: str = eeg_key
        self.data_pairs: List[Tuple[str, str, str]] = []

        mat_files: List[str] = []
        excel_files: dict[str, str] = {}
        # --- scan once
        for root, _, files in os.walk(self.root_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                if fname.endswith(".mat"):
                    mat_files.append(full_path)
                elif fname.endswith(".xlsx") and fname.startswith("yn_"):
                    base = os.path.splitext(fname)[0].replace("yn_", "")
                    excel_files[base] = full_path

        # --- pair: "<base>_CORTADO_.mat" with "yn_<base>.xlsx"
        for mat_path in mat_files:
            file_name = os.path.basename(mat_path)
            if "_CORTADO_" not in file_name:
                continue

            base_name = file_name.replace("_CORTADO_", "_").replace(".mat", "")
            excel_path = excel_files.get(base_name)
            if excel_path:
                self.data_pairs.append((mat_path, excel_path, base_name))

        if not self.data_pairs:
            raise ValueError("No valid (.mat, .xlsx) pairs were found in the tree.")

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        mat_path, excel_path, base_name = self.data_pairs[idx]

        # --- load EEG from .mat
        mat_data = loadmat(mat_path)
        if self.eeg_key not in mat_data:
            raise KeyError(f"Key '{self.eeg_key}' not found in {mat_path}")

        eeg: Tensor = torch.as_tensor(
            mat_data[self.eeg_key]
        ).float()  # (N, T) or (N, ..., T)
        if eeg.ndim < 2:
            raise ValueError(
                f"EEG array must be at least 2D. Got shape {tuple(eeg.shape)} in {mat_path}"
            )

        # --- load behavior from Excel
        df = pd.read_excel(excel_path)

        # Required columns
        if "Missing" not in df.columns:
            raise KeyError(f"Column 'Missing' not found in {excel_path}")
        if "Inhibition" not in df.columns:
            raise KeyError(f"Column 'Inhibition' not found in {excel_path}")

        # Align lengths: Excel rows should match EEG trials along dim 0
        if len(df) != eeg.shape[0]:
            raise ValueError(
                f"Length mismatch for {base_name}: EEG={eeg.shape[0]} vs Excel={len(df)}"
            )

        # Build boolean mask using pure-Python → torch (no numpy API in user code)
        first_col = str(df.columns[0])
        mask_list = [
            (int(row[first_col]) == 5) and (int(row["Missing"]) == 1)
            for _, row in df.iterrows()
        ]
        mask: Tensor = torch.tensor(mask_list, dtype=torch.bool)

        # Apply mask to EEG trials (mask is (N,))
        eeg = eeg[mask]

        # Build labels (0/1) from "Inhibition" column, then to LongTensor
        inhib_list = [row["Inhibition"] for _, row in df.iterrows()]
        label_list = [1 if str(v) == "1" or v == 1 else 0 for v in inhib_list]
        label_full = torch.tensor(label_list, dtype=torch.long)
        label = label_full[mask]

        return eeg, label, base_name
