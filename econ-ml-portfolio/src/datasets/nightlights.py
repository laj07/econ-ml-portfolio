"""
VIIRS annual nightlight composite tile dataset for the informal economy experiment.

Loads 256×256 pixel tiles from VIIRS VNL v2 annual composites, paired with
subnational economic activity proxies (World Bank subnational poverty data,
OpenStreetMap settlement labels).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class NightlightTileDataset(Dataset):
    """
    Dataset of VIIRS luminosity tiles with economic proxy targets.

    Expects data_dir to contain:
        tiles/  — .npy files, each (256, 256) float32, radiance in nW/cm²/sr
        labels.csv — columns: filename, log_gdp_pc, region, country

    Args:
        data_dir:   root directory for the nightlights data
        split:      'train', 'val', or 'test'
        transform:  torchvision transforms applied to tile tensor
    """

    # VIIRS VNL v2 global stats (log1p of radiance)
    MEAN = 0.847
    STD  = 1.203

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        transform=None,
    ) -> None:
        self.data_dir  = Path(data_dir)
        self.split     = split
        self.transform = transform
        self.records   = self._load_index()

    def _load_index(self) -> list[dict]:
        import pandas as pd
        label_file = self.data_dir / "labels.csv"
        if not label_file.exists():
            raise FileNotFoundError(
                f"labels.csv not found in {self.data_dir}. "
                "Run: python scripts/download_nightlights.py --out data/nightlights"
            )
        df = pd.read_csv(label_file)
        # Simple deterministic split by hash of filename
        df["_split"] = df["filename"].apply(
            lambda f: ["train", "train", "train", "val", "test"][hash(f) % 5]
        )
        df = df[df["_split"] == self.split].reset_index(drop=True)
        log.info("NightlightTileDataset [%s]: %d tiles", self.split, len(df))
        return df.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec  = self.records[idx]
        path = self.data_dir / "tiles" / rec["filename"]
        tile = np.load(path).astype(np.float32)

        # Log-transform and normalise
        tile = (np.log1p(tile) - self.MEAN) / self.STD
        x    = torch.from_numpy(tile).unsqueeze(0)  # (1, 256, 256)

        if self.transform:
            x = self.transform(x)

        return {
            "x":      x,
            "target": torch.tensor(rec["log_gdp_pc"], dtype=torch.float32),
        }

    # ── Smoke-test synthetic ──────────────────────────────────────────────────

    @staticmethod
    def synthetic(n: int = 32, tile_size: int = 64) -> "NightlightTileDataset":
        """Return a tiny synthetic dataset for smoke testing."""
        rng = np.random.default_rng(0)

        class _SyntheticDS(Dataset):
            def __len__(self):
                return n
            def __getitem__(self, idx):
                x      = torch.from_numpy(rng.random((1, tile_size, tile_size)).astype(np.float32))
                target = torch.tensor(rng.normal(8.0, 1.5), dtype=torch.float32)
                return {"x": x, "target": target}

        return _SyntheticDS()
