from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

HEADERS = {"X-API-Key": "test-key"}
BASE = datetime(2026, 10, 6, 14, 0)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app import database, main

    engine = create_engine(f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False})
    TestSession = sessionmaker(bind=engine)
    monkeypatch.setattr(main, "engine", engine)

    def override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[database.get_db] = override_db
    with TestClient(main.app) as c:
        yield c
    main.app.dependency_overrides.clear()


def tx(**overrides):
    body = {
        "account_id": "acc-1",
        "amount": 20.0,
        "currency": "gbp",
        "merchant_category": "Groceries",
        "country": "gb",
        "timestamp": BASE.isoformat(),
    }
    body.update(overrides)
    return body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_requires_api_key(client):
    assert client.post("/transactions", json=tx()).status_code == 401
    assert client.post("/transactions", json=tx(), headers={"X-API-Key": "wrong"}).status_code == 401


def test_rejects_invalid_input(client):
    assert client.post("/transactions", json=tx(amount=-5), headers=HEADERS).status_code == 422
    assert client.post("/transactions", json=tx(country="GBR"), headers=HEADERS).status_code == 422


def test_scores_and_stores_transaction(client):
    r = client.post("/transactions", json=tx(), headers=HEADERS)
    assert r.status_code == 201
    body = r.json()
    assert body["decision"] == "approve"
    assert body["currency"] == "GBP" and body["merchant_category"] == "groceries"

    fetched = client.get(f"/transactions/{body['id']}", headers=HEADERS)
    assert fetched.status_code == 200
    assert fetched.json() == body


def test_unknown_transaction_returns_404(client):
    assert client.get("/transactions/999", headers=HEADERS).status_code == 404


def test_suspicious_pattern_is_flagged_and_filterable(client):
    for day in range(1, 11):  # build a normal history
        client.post("/transactions", json=tx(timestamp=(BASE - timedelta(days=day)).isoformat()), headers=HEADERS)

    night = BASE.replace(hour=2)
    for m in range(5, 0, -1):  # rapid burst
        client.post("/transactions", json=tx(timestamp=(night - timedelta(minutes=m)).isoformat()), headers=HEADERS)

    r = client.post(
        "/transactions",
        json=tx(amount=900, merchant_category="crypto", country="NG", timestamp=night.isoformat()),
        headers=HEADERS,
    )
    assert r.json()["decision"] in {"review", "decline"}
    assert r.json()["reasons"]

    flagged = client.get("/transactions", params={"account_id": "acc-1", "decision": r.json()["decision"]}, headers=HEADERS)
    assert r.json()["id"] in [t["id"] for t in flagged.json()]
