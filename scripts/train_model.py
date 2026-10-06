"""Train the fraud model on synthetic data and save it to model/model.joblib.

Real fraud labels aren't publicly available at transaction level, so this generates a
synthetic dataset where fraud is rare (~2%) and more likely when risk signals combine.
The point is the pipeline (features -> model -> evaluation -> served by the API), not the data.

Usage: python scripts/train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "model" / "model.joblib"
N = 50_000
SEED = 42


def generate(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    amount_ratio = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    tx_last_10_min = rng.poisson(0.4, size=n)
    new_country = rng.random(n) < 0.05
    night_time = rng.random(n) < 0.08
    high_risk = rng.random(n) < 0.06

    # Fraudsters: inject a minority with exaggerated signals.
    fraud_like = rng.random(n) < 0.03
    amount_ratio[fraud_like] *= rng.uniform(3, 20, fraud_like.sum())
    tx_last_10_min[fraud_like] += rng.poisson(3, fraud_like.sum())
    new_country[fraud_like] |= rng.random(fraud_like.sum()) < 0.5
    night_time[fraud_like] |= rng.random(fraud_like.sum()) < 0.3

    logit = (
        -6.0
        + 0.25 * np.minimum(amount_ratio, 50)
        + 0.6 * tx_last_10_min
        + 1.5 * new_country
        + 0.8 * night_time
        + 1.2 * high_risk
    )
    y = rng.random(n) < 1 / (1 + np.exp(-logit))

    X = np.column_stack(
        [np.minimum(amount_ratio, 50), tx_last_10_min, new_country, night_time, high_risk]
    ).astype(float)
    return X, y.astype(int)


def main() -> None:
    rng = np.random.default_rng(SEED)
    X, y = generate(N, rng)
    print(f"Generated {N} transactions, fraud rate {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    print(f"ROC AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, proba >= 0.5, target_names=["legit", "fraud"], digits=3))

    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved model to {OUT}")


if __name__ == "__main__":
    main()
