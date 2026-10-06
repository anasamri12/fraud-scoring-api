from datetime import datetime, timedelta
from pathlib import Path

from app.features import build_features
from app.scoring import Scorer, rule_score

NOW = datetime(2026, 10, 6, 14, 0)


def history(n: int, amount: float = 20.0, country: str = "GB", spacing_min: int = 60 * 24):
    return [(amount, country, NOW - timedelta(minutes=spacing_min * (i + 1))) for i in range(n)]


def rules_only() -> Scorer:
    return Scorer(model_path=Path("does-not-exist.joblib"))


def test_normal_transaction_is_approved():
    f = build_features(25.0, "groceries", "GB", NOW, history(10))
    result = rules_only().score(f)
    assert result.decision == "approve"
    assert result.reasons == []


def test_large_amount_flagged_with_reason():
    f = build_features(500.0, "electronics", "GB", NOW, history(10))
    score, reasons = rule_score(f)
    assert score >= 0.45
    assert any("account average" in r for r in reasons)


def test_velocity_counts_only_recent_transactions():
    recent = [(20.0, "GB", NOW - timedelta(minutes=m)) for m in (1, 2, 3, 4, 5)]
    f = build_features(20.0, "groceries", "GB", NOW, recent + history(5))
    assert f.tx_last_10_min == 5


def test_new_country_needs_history():
    assert build_features(20.0, "groceries", "FR", NOW, []).new_country is False
    assert build_features(20.0, "groceries", "FR", NOW, history(3)).new_country is True


def test_combined_signals_are_declined():
    recent = [(20.0, "GB", NOW.replace(hour=2) - timedelta(minutes=m)) for m in (1, 2, 3, 4, 5)]
    f = build_features(900.0, "crypto", "NG", NOW.replace(hour=2), recent + history(10))
    result = rules_only().score(f)
    assert result.decision == "decline"
    assert len(result.reasons) >= 4


def test_score_is_capped_at_one():
    night = NOW.replace(hour=3)
    recent = [(20.0, "GB", night - timedelta(minutes=m)) for m in range(1, 8)]
    f = build_features(5000.0, "gambling", "US", night, recent + history(10))
    assert f.tx_last_10_min == 7
    score, _ = rule_score(f)
    assert score == 1.0
