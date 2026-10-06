from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Decision = Literal["approve", "review", "decline"]


class TransactionIn(BaseModel):
    account_id: str = Field(min_length=1, max_length=64)
    amount: float = Field(gt=0, le=1_000_000)
    currency: str = Field(min_length=3, max_length=3)
    merchant_category: str = Field(min_length=1, max_length=32)
    country: str = Field(min_length=2, max_length=2, description="ISO 3166-1 alpha-2 code")
    timestamp: datetime

    @field_validator("currency", "country")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.upper()

    @field_validator("merchant_category")
    @classmethod
    def lower(cls, v: str) -> str:
        return v.lower()


class TransactionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    account_id: str
    amount: float
    currency: str
    merchant_category: str
    country: str
    timestamp: datetime
    risk_score: float
    decision: Decision
    reasons: list[str]

    @field_validator("reasons", mode="before")
    @classmethod
    def split_reasons(cls, v):
        if isinstance(v, str):
            return [r for r in v.split(";") if r]
        return v
