"""Dataset OpenContrails : charge un record et produit une entrée Ash RGB + le masque cible.

Représentation d'entrée : schéma Ash RGB (bandes 11/13/14/15), cohérent avec
l'exploration menée dans notebooks/01_data_exploration.ipynb. Seule la frame
annotée (TARGET_FRAME) est utilisée — pas de contexte temporel à ce stade
(voir Phase 3 du plan de cadrage pour l'extension multi-frame).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

TARGET_FRAME = 4

_T13_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


def normalize_range(data: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def ash_color_scheme(band11: np.ndarray, band13: np.ndarray, band14: np.ndarray, band15: np.ndarray) -> np.ndarray:
    """Retourne un tenseur (H, W, 3) dans [0, 1]."""
    r = normalize_range(band15 - band13, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band13, _T13_BOUNDS)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)


def load_ash_rgb(record_dir: Path, frame: int = TARGET_FRAME) -> np.ndarray:
    band11 = np.load(record_dir / "band_11.npy")[:, :, frame]
    band13 = np.load(record_dir / "band_13.npy")[:, :, frame]
    band14 = np.load(record_dir / "band_14.npy")[:, :, frame]
    band15 = np.load(record_dir / "band_15.npy")[:, :, frame]
    return ash_color_scheme(band11, band13, band14, band15)


class ContrailAshRGBDataset(Dataset):
    """Un exemple = (image Ash RGB (3, H, W) float32, masque (1, H, W) float32)."""

    def __init__(self, split_dir: Path):
        self.split_dir = Path(split_dir)
        self.record_dirs = sorted(p for p in self.split_dir.iterdir() if p.is_dir())
        if not self.record_dirs:
            raise FileNotFoundError(f"Aucun record trouvé dans {self.split_dir}")

    def __len__(self) -> int:
        return len(self.record_dirs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record_dir = self.record_dirs[idx]

        ash_rgb = load_ash_rgb(record_dir)  # (H, W, 3)
        image = torch.from_numpy(ash_rgb.transpose(2, 0, 1)).float()  # (3, H, W)

        mask = np.load(record_dir / "human_pixel_masks.npy").squeeze(-1)  # (H, W)
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        return image, mask, record_dir.name
