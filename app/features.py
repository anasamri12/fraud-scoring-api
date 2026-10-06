"""Turn a transaction plus the account's history into the features used for scoring."""

from dataclasses import dataclass
from datetime import datetime, timedelta

HIGH_RISK_CATEGORIES = {"gambling", "crypto", "gift_cards", "money_transfer"}
VELOCITY_WINDOW = timedelta(minutes=10)


@dataclass
class Features:
    amount_ratio: float  # amount / account's average amount (1.0 when no history)
    tx_last_10_min: int  # earlier transactions from the same account in the window
    new_country: bool  # country never seen for this account before
    night_time: bool  # between 00:00 and 05:00
    high_risk_category: bool
    has_history: bool

    def as_vector(self) -> list[float]:
        return [
            min(self.amount_ratio, 50.0),
            float(self.tx_last_10_min),
            float(self.new_country),
            float(self.night_time),
            float(self.high_risk_category),
        ]


def build_features(
    amount: float,
    merchant_category: str,
    country: str,
    timestamp: datetime,
    history: list[tuple[float, str, datetime]],
) -> Features:
    """`history` is (amount, country, timestamp) for the account's earlier transactions."""
    has_history = len(history) > 0
    avg = sum(h[0] for h in history) / len(history) if has_history else amount
    seen_countries = {h[1] for h in history}
    recent = [h for h in history if timedelta(0) <= timestamp - h[2] <= VELOCITY_WINDOW]

    return Features(
        amount_ratio=amount / avg if avg > 0 else 1.0,
        tx_last_10_min=len(recent),
        new_country=has_history and country not in seen_countries,
        night_time=0 <= timestamp.hour < 5,
        high_risk_category=merchant_category in HIGH_RISK_CATEGORIES,
        has_history=has_history,
    )
