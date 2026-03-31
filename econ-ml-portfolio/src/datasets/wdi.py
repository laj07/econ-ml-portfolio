"""
World Bank World Development Indicators (WDI) loader for GDP nowcasting.

Downloads GDP growth and alt-data proxies (Google Trends via pytrends,
VIIRS nightlights annual) and assembles a panel dataset suitable for
pytorch-forecasting's TimeSeriesDataSet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# WDI indicator codes we use
WDI_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",      # GDP growth (annual %)
    "FP.CPI.TOTL.ZG":    "inflation",        # CPI inflation (annual %)
    "NE.TRD.GNFS.ZS":    "trade_openness",   # Trade (% of GDP)
    "SL.UEM.TOTL.ZS":    "unemployment",     # Unemployment rate (%)
    "NY.GDP.PCAP.PP.KD":  "gdp_per_capita",  # GDP per capita PPP (constant 2017 USD)
}

# Countries with reasonable WDI coverage — expand as needed
COUNTRY_ISO3 = [
    "IND", "BRA", "ZAF", "IDN", "NGA", "KEN", "BGD",
    "PAK", "VNM", "ETH", "PHL", "MEX", "TUR", "EGY",
]


class WDIPanel:
    """
    Downloads and caches World Bank WDI data for a panel of countries.

    Args:
        countries: list of ISO-3 country codes
        indicators: dict mapping WDI code → column name
        start_year: first year
        end_year:   last year
        cache_dir:  where to save/load the parquet cache
    """

    def __init__(
        self,
        countries: list[str] = COUNTRY_ISO3,
        indicators: dict[str, str] = WDI_INDICATORS,
        start_year: int = 2000,
        end_year: int = 2023,
        cache_dir: str | Path = "data/wdi",
    ) -> None:
        self.countries   = countries
        self.indicators  = indicators
        self.start_year  = start_year
        self.end_year    = end_year
        self.cache_dir   = Path(cache_dir)
        self._panel: pd.DataFrame | None = None

    def load(self, force_download: bool = False) -> pd.DataFrame:
        cache_path = self.cache_dir / "panel.parquet"

        if cache_path.exists() and not force_download:
            log.info("Loading WDI panel from cache: %s", cache_path)
            self._panel = pd.read_parquet(cache_path)
            return self._panel

        try:
            import wbgapi as wb
        except ImportError as exc:
            raise ImportError("pip install wbgapi") from exc

        log.info("Downloading WDI indicators from World Bank API …")
        dfs = []
        for code, col_name in self.indicators.items():
            try:
                df = wb.data.DataFrame(
                    code,
                    economy=self.countries,
                    time=range(self.start_year, self.end_year + 1),
                    numericTimeKeys=True,
                ).reset_index()
                df = df.melt(id_vars=["economy"], var_name="year", value_name=col_name)
                df["year"] = df["year"].astype(int)
                dfs.append(df)
            except Exception as e:
                log.warning("Failed to fetch %s: %s", code, e)

        panel = dfs[0]
        for df in dfs[1:]:
            panel = panel.merge(df, on=["economy", "year"], how="outer")

        panel = panel.rename(columns={"economy": "country"})
        panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache_path, index=False)
        log.info("WDI panel saved → %s  (%d rows)", cache_path, len(panel))

        self._panel = panel
        return panel

    def to_timeseries_dataset(self, min_obs: int = 10):
        """
        Convert panel to a pytorch_forecasting TimeSeriesDataSet.
        Requires pytorch-forecasting to be installed.
        """
        try:
            from pytorch_forecasting import TimeSeriesDataSet
        except ImportError as exc:
            raise ImportError("pip install pytorch-forecasting") from exc

        df = self.load().dropna(subset=["gdp_growth"])

        # Require at least min_obs observations per country
        counts = df.groupby("country")["gdp_growth"].count()
        valid  = counts[counts >= min_obs].index
        df     = df[df["country"].isin(valid)].copy()

        # time_idx: integer index starting at 0 per country
        df["time_idx"] = df.groupby("country").cumcount()

        feature_cols = [c for c in df.columns
                        if c not in ("country", "year", "gdp_growth", "time_idx")]

        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="gdp_growth",
            group_ids=["country"],
            min_encoder_length=8,
            max_encoder_length=24,
            min_prediction_length=1,
            max_prediction_length=4,
            time_varying_unknown_reals=["gdp_growth"] + feature_cols,
            target_normalizer=None,
        )
        return dataset

    @staticmethod
    def synthetic_panel(n_countries: int = 5, n_years: int = 20) -> pd.DataFrame:
        """Tiny synthetic panel for smoke testing."""
        rng = np.random.default_rng(42)
        rows = []
        for c in [f"C{i}" for i in range(n_countries)]:
            for y in range(2000, 2000 + n_years):
                rows.append({
                    "country":       c,
                    "year":          y,
                    "gdp_growth":    rng.normal(3.0, 2.0),
                    "inflation":     rng.uniform(1, 10),
                    "trade_openness":rng.uniform(20, 80),
                    "unemployment":  rng.uniform(3, 15),
                })
        return pd.DataFrame(rows)
