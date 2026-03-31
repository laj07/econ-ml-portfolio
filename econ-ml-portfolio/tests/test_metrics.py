"""Unit tests for evaluation metrics."""
import numpy as np
import pytest
from src.utils.metrics import ndcg_at_k, mae, rmse, mape, pearson_r, macro_f1, mean_reciprocal_rank


def test_ndcg_perfect():
    y_true  = np.array([1, 0, 1, 0, 1])
    y_score = np.array([5, 1, 4, 2, 3])
    score   = ndcg_at_k(y_true, y_score, k=3)
    assert score == pytest.approx(1.0, abs=1e-4)


def test_ndcg_worst():
    y_true  = np.array([1, 0, 0])
    y_score = np.array([1, 2, 3])   # worst possible ranking
    score   = ndcg_at_k(y_true, y_score, k=3)
    assert score < 0.5


def test_ndcg_range():
    rng = np.random.default_rng(42)
    for _ in range(50):
        y_true  = (rng.random(20) > 0.7).astype(float)
        y_score = rng.random(20)
        score   = ndcg_at_k(y_true, y_score, k=10)
        assert 0.0 <= score <= 1.0


def test_mrr():
    y_true  = np.array([0, 0, 1, 0])
    y_score = np.array([1, 2, 4, 3])
    mrr = mean_reciprocal_rank(y_true, y_score)
    assert mrr == pytest.approx(1.0)   # rank-1 hit


def test_mae_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([1.0, 0.0])
    assert rmse(y_true, y_pred) == pytest.approx(1.0)


def test_mape():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    result = mape(y_true, y_pred)
    # (10/100 + 20/200) / 2 * 100 = 10.0
    assert result == pytest.approx(10.0, rel=1e-3)


def test_pearson_perfect():
    x = np.arange(10, dtype=float)
    assert pearson_r(x, x)      == pytest.approx(1.0)
    assert pearson_r(x, -x)     == pytest.approx(-1.0)


def test_macro_f1_perfect():
    y = [0, 1, 2, 0, 1, 2]
    assert macro_f1(y, y) == pytest.approx(1.0)


def test_macro_f1_range():
    y_true = [0, 1, 2, 1, 0]
    y_pred = [0, 2, 1, 1, 0]
    score  = macro_f1(y_true, y_pred)
    assert 0.0 <= score <= 1.0
