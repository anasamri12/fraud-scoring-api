import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.features import build_features
from app.models import Transaction
from app.schemas import Decision, TransactionIn, TransactionOut
from app.scoring import Scorer

HISTORY_LIMIT = 200  # most recent transactions used for an account's profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    app.state.scorer = Scorer()
    yield


app = FastAPI(
    title="Transaction Fraud-Scoring API",
    description="Scores card transactions for fraud risk using rules plus an ML model.",
    version="1.0.0",
    lifespan=lifespan,
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("API_KEY", "dev-key")
    if x_api_key != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.scorer.model is not None}


@app.post(
    "/transactions",
    response_model=TransactionOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def score_transaction(tx: TransactionIn, db: Session = Depends(get_db)):
    rows = db.execute(
        select(Transaction.amount, Transaction.country, Transaction.timestamp)
        .where(Transaction.account_id == tx.account_id, Transaction.timestamp <= tx.timestamp)
        .order_by(Transaction.timestamp.desc())
        .limit(HISTORY_LIMIT)
    ).all()
    history = [(r.amount, r.country, r.timestamp) for r in rows]

    features = build_features(tx.amount, tx.merchant_category, tx.country, tx.timestamp, history)
    result = app.state.scorer.score(features)

    record = Transaction(
        **tx.model_dump(),
        risk_score=result.risk_score,
        decision=result.decision,
        reasons=";".join(result.reasons),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get(
    "/transactions/{tx_id}",
    response_model=TransactionOut,
    dependencies=[Depends(require_api_key)],
)
def get_transaction(tx_id: int, db: Session = Depends(get_db)):
    record = db.get(Transaction, tx_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Transaction not found")
    return record


@app.get(
    "/transactions",
    response_model=list[TransactionOut],
    dependencies=[Depends(require_api_key)],
)
def list_transactions(
    account_id: str | None = None,
    decision: Decision | None = None,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    query = select(Transaction).order_by(Transaction.timestamp.desc()).limit(limit)
    if account_id:
        query = query.where(Transaction.account_id == account_id)
    if decision:
        query = query.where(Transaction.decision == decision)
    return db.scalars(query).all()
