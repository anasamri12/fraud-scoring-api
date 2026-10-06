"""Combine transparent rules with an optional ML model into one risk score and decision."""

from dataclasses import dataclass, field
from pathlib import Path

import joblib

from app.features import Features

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"

REVIEW_THRESHOLD = 0.4
DECLINE_THRESHOLD = 0.75
RULE_WEIGHT = 0.6  # share of the final score from rules when a model is loaded


@dataclass
class ScoreResult:
    risk_score: float
    decision: str
    reasons: list[str] = field(default_factory=list)


def rule_score(f: Features) -> tuple[float, list[str]]:
    """Each triggered rule adds points; the total is capped at 1.0. Reasons explain the score."""
    score, reasons = 0.0, []

    if f.amount_ratio >= 10:
        score += 0.45
        reasons.append(f"amount is {f.amount_ratio:.1f}x the account average")
    elif f.amount_ratio >= 3:
        score += 0.2
        reasons.append(f"amount is {f.amount_ratio:.1f}x the account average")

    if f.tx_last_10_min >= 5:
        score += 0.35
        reasons.append(f"{f.tx_last_10_min} transactions in the last 10 minutes")
    elif f.tx_last_10_min >= 3:
        score += 0.15
        reasons.append(f"{f.tx_last_10_min} transactions in the last 10 minutes")

    if f.new_country:
        score += 0.25
        reasons.append("first transaction from this country")
    if f.high_risk_category:
        score += 0.2
        reasons.append("high-risk merchant category")
    if f.night_time:
        score += 0.1
        reasons.append("made between 00:00 and 05:00")

    return min(round(score, 3), 1.0), reasons


class Scorer:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = joblib.load(model_path) if model_path.exists() else None

    def score(self, f: Features) -> ScoreResult:
        rules, reasons = rule_score(f)
        if self.model is not None:
            prob = float(self.model.predict_proba([f.as_vector()])[0][1])
            final = RULE_WEIGHT * rules + (1 - RULE_WEIGHT) * prob
            if prob >= 0.8:
                reasons.append(f"model fraud probability {prob:.2f}")
        else:
            final = rules

        final = round(final, 3)
        if final >= DECLINE_THRESHOLD:
            decision = "decline"
        elif final >= REVIEW_THRESHOLD:
            decision = "review"
        else:
            decision = "approve"
        return ScoreResult(final, decision, reasons)
