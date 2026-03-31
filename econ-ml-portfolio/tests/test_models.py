"""Unit tests for model architectures (CPU, no real data)."""
import pytest
import numpy as np
import torch


# ── GNN ──────────────────────────────────────────────────────────────────────

def test_gnn_forward():
    try:
        from src.models.gnn import ProductSpaceGNN, ECIRegressor, LinkPredictor
        from torch_geometric.data import Data
    except ImportError:
        pytest.skip("torch-geometric not installed")

    n_nodes, in_ch, out_ch = 30, 30, 16
    x          = torch.eye(n_nodes)
    edge_index = torch.randint(0, n_nodes, (2, 50))
    country_mask = torch.zeros(n_nodes, dtype=torch.bool)
    country_mask[:10] = True

    gnn  = ProductSpaceGNN(in_channels=in_ch, hidden_channels=32, out_channels=out_ch, num_layers=2)
    emb  = gnn(x, edge_index)
    assert emb.shape == (n_nodes, out_ch), f"Expected ({n_nodes}, {out_ch}), got {emb.shape}"

    reg  = ECIRegressor(gnn, embedding_dim=out_ch)
    pred = reg(x, edge_index, country_mask)
    assert pred.shape == (10,), f"Expected (10,), got {pred.shape}"

    lp   = LinkPredictor(embedding_dim=out_ch)
    src  = torch.randint(0, n_nodes, (20,))
    dst  = torch.randint(0, n_nodes, (20,))
    out  = lp(emb, src, dst)
    assert out.shape == (20,)


# ── NLP ───────────────────────────────────────────────────────────────────────

def test_nlp_forward():
    pytest.importorskip("transformers")
    from src.models.nlp import CentralBankClassifier
    from transformers import AutoTokenizer

    model_name = "prajjwal1/bert-tiny"   # tiny BERT for fast CI
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = CentralBankClassifier(model_name=model_name, num_labels=3)
    except Exception:
        pytest.skip("Could not download tiny BERT model (no network?)")

    enc = tokenizer(["Rates will be raised.", "Policy remains accommodative."],
                    padding=True, return_tensors="pt")
    logits = model(enc["input_ids"], enc["attention_mask"])
    assert logits.shape == (2, 3)


# ── LSTM ──────────────────────────────────────────────────────────────────────

def test_lstm_forward():
    from src.models.lstm import GDPNowcastLSTM
    model = GDPNowcastLSTM(n_features=5, hidden_size=32, num_layers=2)
    x     = torch.randn(8, 10, 5)   # batch=8, seq=10, features=5
    out   = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_vsn_forward():
    from src.models.lstm import VariableSelectionNetwork
    vsn  = VariableSelectionNetwork(n_features=5, hidden_size=16)
    x    = torch.randn(4, 8, 5)
    sel, weights = vsn(x)
    assert sel.shape     == (4, 8, 16)
    assert weights.shape == (4, 8, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 8), atol=1e-5)


# ── CNN ───────────────────────────────────────────────────────────────────────

def test_cnn_forward():
    pytest.importorskip("torchvision")
    from src.models.cnn import NightlightRegressor
    model = NightlightRegressor(in_channels=1, pretrained=False, dropout=0.1)
    x     = torch.randn(4, 1, 64, 64)
    out   = model(x)
    assert out.shape == (4,)


# ── Embeddings ────────────────────────────────────────────────────────────────

def test_mismatch_analyser():
    from src.models.embeddings import MismatchAnalyser
    rng          = np.random.default_rng(0)
    demand_embs  = rng.random((20, 32)).astype(np.float32)
    supply_embs  = rng.random((30, 32)).astype(np.float32)
    # L2-normalise
    demand_embs /= np.linalg.norm(demand_embs, axis=1, keepdims=True)
    supply_embs /= np.linalg.norm(supply_embs, axis=1, keepdims=True)

    analyser = MismatchAnalyser(
        demand_embs, supply_embs,
        [f"d{i}" for i in range(20)],
        [f"s{i}" for i in range(30)],
    )
    score = analyser.mismatch_score(threshold=0.5)
    assert 0.0 <= score <= 1.0

    orphans = analyser.orphan_demand(top_n=5)
    assert len(orphans) == 5
