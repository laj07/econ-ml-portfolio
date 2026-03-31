"""
Download World Bank WDI macro panel data.

Usage
-----
    python scripts/download_wdi.py --out data/wdi
    python scripts/download_wdi.py --out data/wdi --countries IND BRA ZAF
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("download_wdi")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WDI panel data")
    parser.add_argument("--out", default="data/wdi")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="ISO-3 codes; defaults to the 14 countries in the experiment")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year",   type=int, default=2023)
    parser.add_argument("--force", action="store_true", help="Re-download even if cache exists")
    args = parser.parse_args()

    from src.datasets.wdi import WDIPanel, COUNTRY_ISO3

    countries = args.countries or COUNTRY_ISO3
    panel = WDIPanel(
        countries=countries,
        start_year=args.start_year,
        end_year=args.end_year,
        cache_dir=args.out,
    )
    df = panel.load(force_download=args.force)
    log.info("Panel shape: %s", df.shape)
    log.info("Countries  : %s", sorted(df['country'].unique()))
    log.info("Years      : %d – %d", df['year'].min(), df['year'].max())
    log.info("Saved → %s/panel.parquet", args.out)


if __name__ == "__main__":
    main()
