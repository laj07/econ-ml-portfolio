"""
Download ESCO v1.2 skills taxonomy CSV files.

Usage
-----
    python scripts/download_esco.py --out data/esco
"""
from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("download_esco")

# ESCO v1.2 bulk download (CSV, English) — official European Commission URL
ESCO_DOWNLOAD_URL = (
    "https://esco.ec.europa.eu/en/use-esco/download"
    # Direct CSV package link (may change with new versions):
    # Check https://esco.ec.europa.eu/en/use-esco/download for the latest URL
)

ESCO_CSV_URL = (
    "https://ec.europa.eu/esco/portal/api/taxonomy/download"
    "?selectedVersion=v1.2.0&formats[]=csv&language=en"
)


def download_esco_csv(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "esco_v1.2_csv_en.zip"

    if not zip_path.exists():
        log.info("Downloading ESCO v1.2 CSV package …")
        try:
            r = requests.get(ESCO_CSV_URL, stream=True, timeout=60)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            log.info("Downloaded → %s", zip_path)
        except Exception as e:
            log.error("Download failed: %s", e)
            log.info(
                "Manual download: visit %s and download the CSV package for English v1.2\n"
                "Extract to: %s", ESCO_DOWNLOAD_URL, out_dir
            )
            (out_dir / "DOWNLOAD_INSTRUCTIONS.txt").write_text(
                f"Visit {ESCO_DOWNLOAD_URL} and download the CSV bulk download for English v1.2.\n"
                f"Extract the zip contents to {out_dir}/\n"
            )
            return

    # Extract
    log.info("Extracting %s …", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    csv_files = list(out_dir.glob("*.csv"))
    log.info("Extracted %d CSV files:", len(csv_files))
    for f in csv_files:
        log.info("  %s  (%.1f MB)", f.name, f.stat().st_size / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ESCO skills taxonomy")
    parser.add_argument("--out", default="data/esco")
    args = parser.parse_args()
    download_esco_csv(Path(args.out))


if __name__ == "__main__":
    main()
