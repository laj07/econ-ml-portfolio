"""Unit tests for dataset loaders using synthetic / in-memory data."""
import numpy as np
import pytest


def test_oec_synthetic_graph():
    try:
        from src.datasets.oec import OECDataset
    except ImportError:
        pytest.skip("torch-geometric not installed")

    data = OECDataset.synthetic_graph(n_countries=10, n_products=20)
    assert data.x.shape[0] == 30
    assert data.edge_index.shape[0] == 2
    assert data.country_mask.sum() == 10


def test_wdi_synthetic_panel():
    from src.datasets.wdi import WDIPanel
    df = WDIPanel.synthetic_panel(n_countries=3, n_years=10)
    assert len(df) == 30
    assert "gdp_growth" in df.columns
    assert df["country"].nunique() == 3


def test_fomc_synthetic_dataset():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from src.datasets.fomc import FOMCDataset

    try:
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    except Exception:
        pytest.skip("No network / model unavailable")

    ds = FOMCDataset.synthetic(tokenizer, n=16, max_length=32)
    assert len(ds) == 16
    item = ds[0]
    assert "input_ids" in item
    assert "label" in item
    assert item["input_ids"].shape[0] == 32


def test_nightlights_synthetic():
    ds = None
    try:
        from src.datasets.nightlights import NightlightTileDataset
        ds = NightlightTileDataset.synthetic(n=8, tile_size=32)
    except Exception as e:
        pytest.skip(str(e))
    assert len(ds) == 8
    item = ds[0]
    assert item["x"].shape == (1, 32, 32)
    assert item["target"].ndim == 0


def test_esco_synthetic_postings():
    from src.datasets.esco import JobPostingsDataset
    ds = JobPostingsDataset.synthetic(n_postings=20)
    texts = ds.posting_texts()
    assert len(texts) == 20
    skills = ds.required_skills()
    assert len(skills) == 20
    assert all(isinstance(s, list) for s in skills)
