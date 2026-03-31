"""
Download and tile VIIRS annual nighttime lights composites.

Data source: Earth Observation Group, Colorado School of Mines
             https://eogdata.mines.edu/products/vnl/

The full annual composite is ~4 GB. This script downloads a smaller
regional subset (South Asia by default) for tractable experiments.

Usage
-----
    python scripts/download_nightlights.py --out data/nightlights --region south_asia
    python scripts/download_nightlights.py --out data/nightlights --region africa
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("download_nightlights")

# Bounding boxes: (min_lon, min_lat, max_lon, max_lat)
REGIONS = {
    "south_asia":    (60.0,  5.0,  100.0,  40.0),
    "africa":        (-20.0, -35.0,  55.0,  38.0),
    "southeast_asia":(95.0,  -10.0, 141.0,  28.0),
    "latin_america": (-82.0, -56.0,  -34.0,  13.0),
}

VNL_BASE = "https://eogdata.mines.edu/nighttime_light/annual/v21/{year}/"


def print_manual_instructions(out_dir: Path) -> None:
    msg = f"""
VIIRS Nightlights Download Instructions
=========================================
Automated bulk download requires a free NASA Earthdata account.

Manual steps:
1. Create a free account at: https://urs.earthdata.nasa.gov/
2. Go to: https://eogdata.mines.edu/products/vnl/
3. Download the "Annual VNL V2" composite for your years of interest
   - File: VNL_v2_npp_YYYY_global_vcmslcfg_c202205302300.average_masked.tif.gz
4. Extract and place the .tif files in: {out_dir}/raw/
5. Run tile extraction:
   python scripts/download_nightlights.py --out {out_dir} --tile-only

Alternatively, a smaller pre-tiled subset for South Asia (2015-2020) is
available on Zenodo at: https://zenodo.org/record/XXXXXXX (placeholder)
"""
    log.info(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DOWNLOAD_INSTRUCTIONS.txt").write_text(msg)


def tile_existing(raw_dir: Path, out_dir: Path, tile_size: int = 256) -> None:
    """
    Tile raw VIIRS GeoTIFF files into (tile_size × tile_size) numpy arrays.
    Requires rasterio.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        raise ImportError("pip install rasterio numpy")

    import csv

    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    tif_files = list(raw_dir.glob("*.tif"))
    if not tif_files:
        log.warning("No .tif files found in %s", raw_dir)
        return

    records = []
    for tif_path in tif_files:
        year = tif_path.stem.split("_")[4] if "_" in tif_path.stem else "unknown"
        log.info("Tiling %s …", tif_path.name)

        with rasterio.open(tif_path) as src:
            W, H = src.width, src.height
            for row in range(0, H - tile_size + 1, tile_size):
                for col in range(0, W - tile_size + 1, tile_size):
                    window = Window(col, row, tile_size, tile_size)
                    tile   = src.read(1, window=window).astype(np.float32)
                    if tile.max() < 0.1:
                        continue  # skip dark ocean tiles
                    fname = f"tile_{year}_{row:06d}_{col:06d}.npy"
                    np.save(tiles_dir / fname, tile)
                    records.append({
                        "filename":    fname,
                        "year":        year,
                        "log_gdp_pc":  0.0,   # placeholder — join with WB data
                    })

    with open(out_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "year", "log_gdp_pc"])
        writer.writeheader()
        writer.writerows(records)

    log.info("Tiled %d tiles → %s", len(records), tiles_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / tile VIIRS nightlights")
    parser.add_argument("--out",       default="data/nightlights")
    parser.add_argument("--region",    default="south_asia", choices=list(REGIONS))
    parser.add_argument("--tile-only", action="store_true",
                        help="Skip download; tile raw TIFs already in --out/raw/")
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()

    out = Path(args.out)

    if args.tile_only:
        tile_existing(out / "raw", out, tile_size=args.tile_size)
    else:
        print_manual_instructions(out)


if __name__ == "__main__":
    main()
