# Transaction Fraud-Scoring API

A REST API that scores card transactions for fraud risk in real time. Each transaction is checked against the account's history using explainable rules, combined with a logistic-regression model, and returned with a decision (`approve`, `review` or `decline`) and the reasons behind it.

**Stack:** Python, FastAPI, PostgreSQL, SQLAlchemy, scikit-learn, Pytest, Docker, GitHub Actions

## How it works

```
POST /transactions
   │
   ├─ load the account's recent history from Postgres
   ├─ build features: amount vs account average, transactions in the last 10 min,
   │                  first time in this country, night-time, high-risk merchant category
   ├─ rules score (0–1) + a reason for every rule that fires
   ├─ model probability (if a trained model is present)
   ├─ final score = 0.6 × rules + 0.4 × model
   └─ decision: approve (< 0.4) · review (0.4–0.75) · decline (≥ 0.75), stored and returned
```

### Design decisions

- **Rules plus a model, not a model alone.** Banks need to explain why a payment was blocked, to customers and to regulators. Rules give every decision a readable reason; the model catches combinations of weaker signals the rules miss.
- **Three outcomes, not two.** A `review` band sends borderline cases to a human instead of declining genuine customers.
- **History-based features.** "£500" means nothing alone; it is suspicious only if the account normally spends £20. Features are computed against the account's own past behaviour.
- **Runs without the model.** If no model file is present, the API falls back to rules only, so the service never goes down because a model failed to load. CI tests both modes.
- **Synthetic training data.** Real labelled fraud data isn't publicly available at transaction level, so `scripts/train_model.py` generates 50,000 synthetic transactions with a ~2% fraud rate. Classes are heavily imbalanced, so the model uses `class_weight="balanced"` and is judged on ROC AUC and fraud recall rather than accuracy.

Model results on the held-out test set (synthetic data): ROC AUC 0.92, fraud recall 0.79, fraud precision 0.23. Low precision is the expected trade-off at this fraud rate: it is cheaper to review a genuine transaction than to miss a fraudulent one, and the rules component explains each flag.

## Endpoints

All `/transactions` endpoints need an `X-API-Key` header.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Service status and whether the model is loaded |
| POST | `/transactions` | Score a transaction, store it, return the decision |
| GET | `/transactions/{id}` | Fetch one scored transaction |
| GET | `/transactions?account_id=&decision=&limit=` | List transactions, filterable by account and decision |

Interactive docs at `http://localhost:8000/docs` once running.

### Example

```bash
curl -X POST http://localhost:8000/transactions \
  -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
  -d '{"account_id": "acc-1", "amount": 900, "currency": "GBP",
       "merchant_category": "crypto", "country": "NG",
       "timestamp": "2026-10-06T02:00:00"}'
```

```json
{
  "id": 16, "account_id": "acc-1", "amount": 900.0, "currency": "GBP",
  "merchant_category": "crypto", "country": "NG", "timestamp": "2026-10-06T02:00:00",
  "risk_score": 1.0, "decision": "decline",
  "reasons": ["amount is 45.0x the account average", "5 transactions in the last 10 minutes",
              "first transaction from this country", "high-risk merchant category",
              "made between 00:00 and 05:00", "model fraud probability 1.00"]
}
```

## Running it

**With Docker (API + Postgres):**

```bash
docker compose up --build
```

**Locally (SQLite):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train_model.py
uvicorn app.main:app --reload
```

**Tests:**

```bash
pytest
```

## Project structure

```
app/
  main.py        API routes and API-key auth
  features.py    feature engineering from account history
  scoring.py     rules, model blending and decision thresholds
  models.py      SQLAlchemy table
  schemas.py     request/response validation (Pydantic)
  database.py    database connection
scripts/train_model.py   synthetic data, training and evaluation
tests/                   unit tests for scoring, integration tests for the API
```

## Possible next steps

- Database migrations with Alembic instead of `create_all`
- Store model version alongside each decision for auditability
- Rate limiting and per-client API keys
- Retrain on labelled outcomes from the `review` queue
