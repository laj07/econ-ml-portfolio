"""
Download OEC / BACI trade flow data.

The full BACI dataset requires a free CEPII registration. This script
either uses the wbgapi trade indicators (no registration) or guides you
through the BACI download.

Usage
-----
    python scripts/download_oec.py --out data/oec
    python scripts/download_oec.py --out data/oec --source wbgapi
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("download_oec")


def download_via_wbgapi(out_dir: Path, start_year: int = 2000, end_year: int = 2022) -> None:
    """
    Download bilateral trade aggregates from World Bank.
    Granularity: country-level exports/imports, not HS product codes.
    Good enough for ECI regression; use BACI for product-level graph.
    """
    try:
        import wbgapi as wb
    except ImportError:
        raise ImportError("pip install wbgapi")

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading trade data from World Bank API …")
    # NE.EXP.GNFS.CD = exports of goods and services (current USD)
    df = wb.data.DataFrame(
        "NE.EXP.GNFS.CD",
        time=range(start_year, end_year + 1),
        numericTimeKeys=True,
    ).reset_index()
    df = df.melt(id_vars=["economy"], var_name="year", value_name="exports_usd")
    df["year"] = df["year"].astype(int)
    df.columns = ["exporter", "year", "exports_usd"]

    out_path = out_dir / "exports_wbgapi.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved → %s  (%d rows)", out_path, len(df))


def print_baci_instructions() -> None:
    msg = """
BACI HS-92 Download Instructions
==================================
BACI provides bilateral trade flows at HS 6-digit product level (~10M rows/year).
It is free but requires a free registration at CEPII.

1. Register at: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
2. Download BACI_HS92_Vyyyymmdd.zip
3. Extract the CSV files to data/oec/
4. Files should be named like: BACI_HS92_Y2019_V202401.csv

Expected columns: t (year), i (exporter ISO numeric), j (importer), k (HS6 product), v (value kUSD), q (quantity tonnes)

The OECDataset loader handles this format automatically.
"""
    print(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OEC / BACI trade data")
    parser.add_argument("--out",    default="data/oec")
    parser.add_argument("--source", default="wbgapi", choices=["wbgapi", "baci"],
                        help="wbgapi = fast country-level; baci = full product-level (requires registration)")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2022)
    args = parser.parse_args()

    if args.source == "wbgapi":
        download_via_wbgapi(Path(args.out), args.start_year, args.end_year)
    else:
        print_baci_instructions()


if __name__ == "__main__":
    main()
