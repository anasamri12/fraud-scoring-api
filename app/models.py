from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    account_id: Mapped[str] = mapped_column(String(64), index=True)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3))
    merchant_category: Mapped[str] = mapped_column(String(32))
    country: Mapped[str] = mapped_column(String(2))
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)

    risk_score: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    reasons: Mapped[str] = mapped_column(Text, default="")  # semicolon-separated
