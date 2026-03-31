"""
Download FOMC meeting minutes, statements, and ECB/BOE speeches.

Usage
-----
    python scripts/download_fomc.py --out data/fomc
    python scripts/download_fomc.py --out data/fomc --sources fomc ecb
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("download_fomc")

HEADERS = {"User-Agent": "EconMLPortfolio/1.0 (research; contact your@email.com)"}
THROTTLE = 1.0   # seconds between requests — be a polite scraper


# ── FOMC ─────────────────────────────────────────────────────────────────────

FOMC_MINUTES_INDEX = "https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm"


def _get(url: str) -> requests.Response:
    time.sleep(THROTTLE)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def download_fomc_minutes(out_dir: Path, start_year: int = 2000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Fetching FOMC minutes index …")

    soup = BeautifulSoup(_get(FOMC_MINUTES_INDEX).text, "lxml")
    links = soup.find_all("a", href=re.compile(r"fomcminutes\d{8}\.htm"))

    log.info("Found %d minute documents", len(links))

    for a in tqdm(links, desc="FOMC minutes"):
        href  = a["href"]
        match = re.search(r"(\d{8})", href)
        if not match:
            continue
        date = match.group(1)
        year = int(date[:4])
        if year < start_year:
            continue

        url  = "https://www.federalreserve.gov" + href
        dest = out_dir / "fomc" / f"minutes_{date}.txt"
        if dest.exists():
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            page = BeautifulSoup(_get(url).text, "lxml")
            # Main content in <div id="article"> or largest <p> block
            article = page.find("div", id="article") or page.find("div", class_="col-xs-12")
            text    = article.get_text(separator="\n") if article else page.get_text(separator="\n")
            dest.write_text(text.strip(), encoding="utf-8")
        except Exception as e:
            log.warning("Failed %s: %s", url, e)

    log.info("FOMC minutes done → %s", out_dir / "fomc")


# ── ECB ───────────────────────────────────────────────────────────────────────

ECB_SPEECHES_JSON = "https://www.ecb.europa.eu/press/key/date/html/index.en.html"


def download_ecb_speeches(out_dir: Path, max_pages: int = 5) -> None:
    """Download a sample of ECB speeches from the public archive."""
    dest_dir = out_dir / "ecb"
    dest_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching ECB speech index …")
    soup  = BeautifulSoup(_get(ECB_SPEECHES_JSON).text, "lxml")
    links = soup.find_all("a", href=re.compile(r"/press/key/date/\d{4}/html/"))

    log.info("Found %d ECB speech links (sampling first %d pages)", len(links), max_pages)
    for a in tqdm(links[:max_pages * 20], desc="ECB speeches"):
        href  = a.get("href", "")
        url   = "https://www.ecb.europa.eu" + href
        fname = href.rstrip("/").split("/")[-1] + ".txt"
        dest  = dest_dir / fname
        if dest.exists():
            continue
        try:
            page = BeautifulSoup(_get(url).text, "lxml")
            body = page.find("div", class_="ecb-publicationsBulletinArticle") or page.find("main")
            text = body.get_text(separator="\n") if body else page.get_text(separator="\n")
            dest.write_text(text.strip(), encoding="utf-8")
        except Exception as e:
            log.warning("Failed %s: %s", url, e)

    log.info("ECB speeches done → %s", dest_dir)


# ── BOE ───────────────────────────────────────────────────────────────────────

def download_boe_minutes(out_dir: Path) -> None:
    """
    Download Bank of England MPC minutes from the public archive.
    Full scraping is complex; this downloads the index and leaves a note.
    """
    dest_dir = out_dir / "boe"
    dest_dir.mkdir(parents=True, exist_ok=True)

    note = (
        "BOE MPC minutes are available at:\n"
        "https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes\n\n"
        "Download the PDFs manually and place them in this directory,\n"
        "then convert with: for f in *.pdf; do pdftotext $f ${f%.pdf}.txt; done\n"
    )
    (dest_dir / "DOWNLOAD_INSTRUCTIONS.txt").write_text(note)
    log.info("BOE instructions written → %s", dest_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download central bank documents")
    parser.add_argument("--out", default="data/fomc", help="Output directory")
    parser.add_argument(
        "--sources", nargs="+", default=["fomc", "ecb", "boe"],
        choices=["fomc", "ecb", "boe"],
    )
    parser.add_argument("--start-year", type=int, default=2000)
    args = parser.parse_args()

    out = Path(args.out)
    if "fomc" in args.sources:
        download_fomc_minutes(out, start_year=args.start_year)
    if "ecb" in args.sources:
        download_ecb_speeches(out)
    if "boe" in args.sources:
        download_boe_minutes(out)

    log.info("All downloads complete → %s", out)


if __name__ == "__main__":
    main()
