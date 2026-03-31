"""
OEC / BACI trade-flow dataset loader for the economic-complexity experiment.

Builds a PyG HeteroData / bipartite graph from bilateral trade data:
  - Nodes: countries, HS-product categories
  - Edges: RCA > 1 (country exports product competitively)
  - Node features: log-export volume aggregates, region one-hot, income-group one-hot
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# HS-92 sections (broad product groups)
HS_SECTIONS = {
    "live_animals": range(1,  6),
    "vegetables":   range(6,  15),
    "foodstuffs":   range(15, 25),
    "minerals":     range(25, 28),
    "chemicals":    range(28, 39),
    "plastics":     range(39, 41),
    "leather":      range(41, 44),
    "wood":         range(44, 50),
    "textiles":     range(50, 64),
    "footwear":     range(64, 68),
    "metals":       range(72, 84),
    "machinery":    range(84, 86),
    "electronics":  range(85, 86),
    "transport":    range(86, 90),
    "instruments":  range(90, 93),
}


class OECDataset:
    """
    Loads BACI / OEC bilateral trade data and computes RCA-based edges.

    Args:
        data_dir: directory containing baci_hs92_*.csv or oec_*.parquet
        year:     single year to load (or None = latest available)
        min_rca:  RCA threshold for including an edge (default 1.0)
    """

    def __init__(
        self,
        data_dir: str | Path,
        year: int | None = None,
        min_rca: float = 1.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.year     = year
        self.min_rca  = min_rca

        self._trade: pd.DataFrame | None = None
        self._rca:   pd.DataFrame | None = None

    # ── Loaders ──────────────────────────────────────────────────────────────

    def load_trade(self) -> pd.DataFrame:
        """Load bilateral trade data. Expects columns: year, exporter, product, value."""
        files = sorted(self.data_dir.glob("*.parquet")) or \
                sorted(self.data_dir.glob("*.csv"))

        if not files:
            raise FileNotFoundError(
                f"No trade data found in {self.data_dir}. "
                "Run: python scripts/download_oec.py --out data/oec"
            )

        dfs = []
        for f in files:
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            dfs.append(df)

        trade = pd.concat(dfs, ignore_index=True)
        trade.columns = [c.lower().strip() for c in trade.columns]

        if self.year:
            trade = trade[trade["year"] == self.year]

        self._trade = trade
        log.info("Loaded trade data: %d rows, %d countries, %d products",
                 len(trade),
                 trade["exporter"].nunique(),
                 trade["product"].nunique())
        return trade

    # ── RCA computation ───────────────────────────────────────────────────────

    def compute_rca(self) -> pd.DataFrame:
        """
        Balassa's Revealed Comparative Advantage.

        RCA_{c,p} = (X_{c,p} / X_c) / (X_p / X_world)
        """
        if self._trade is None:
            self.load_trade()

        trade = self._trade.groupby(["exporter", "product"])["value"].sum().reset_index()

        total_by_country = trade.groupby("exporter")["value"].sum()
        total_by_product = trade.groupby("product")["value"].sum()
        world_total      = trade["value"].sum()

        trade["rca"] = trade.apply(
            lambda r: (r["value"] / total_by_country[r["exporter"]])
                    / (total_by_product[r["product"]] / world_total),
            axis=1,
        )
        self._rca = trade
        return trade

    # ── Graph construction ────────────────────────────────────────────────────

    def build_graph(self):
        """
        Build a PyG Data object from the RCA matrix.

        Returns a torch_geometric.data.Data with:
          x           — node features (country + product nodes concatenated)
          edge_index  — [2, E] COO edges for RCA > min_rca
          edge_attr   — log(RCA) weights
          y_eci       — ECI targets (NaN for product nodes)
          country_mask — boolean mask for country nodes
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:
            raise ImportError("torch-geometric required.") from exc

        rca = self.compute_rca()
        edges = rca[rca["rca"] >= self.min_rca]

        countries = sorted(edges["exporter"].unique())
        products  = sorted(edges["product"].unique())
        n_c, n_p  = len(countries), len(products)

        c_idx = {c: i for i, c in enumerate(countries)}
        p_idx = {p: i + n_c for i, p in enumerate(products)}

        src = [c_idx[r.exporter] for r in edges.itertuples()]
        dst = [p_idx[r.product]  for r in edges.itertuples()]
        weights = np.log1p(edges["rca"].values).astype(np.float32)

        # Simple node features: one-hot position (can be enriched later)
        x = torch.eye(n_c + n_p, dtype=torch.float32)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(weights, dtype=torch.float32)

        country_mask = torch.zeros(n_c + n_p, dtype=torch.bool)
        country_mask[:n_c] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            country_mask=country_mask,
            countries=countries,
            products=products,
        )
        log.info("Graph: %d nodes (%d countries, %d products), %d edges",
                 n_c + n_p, n_c, n_p, edge_index.shape[1])
        return data

    # ── Smoke-test synthetic graph ────────────────────────────────────────────

    @staticmethod
    def synthetic_graph(n_countries: int = 20, n_products: int = 50):
        """Return a tiny random graph for smoke testing (no data download)."""
        import torch
        from torch_geometric.data import Data

        n = n_countries + n_products
        x = torch.eye(n, dtype=torch.float32)

        # Random sparse edges
        rng   = np.random.default_rng(0)
        mask  = rng.random((n_countries, n_products)) < 0.15
        rows, cols = np.where(mask)
        src = rows.tolist()
        dst = (cols + n_countries).tolist()

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.ones(len(src), dtype=torch.float32)

        country_mask = torch.zeros(n, dtype=torch.bool)
        country_mask[:n_countries] = True

        y_eci = torch.randn(n_countries)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            country_mask=country_mask,
            y_eci=y_eci,
        )
