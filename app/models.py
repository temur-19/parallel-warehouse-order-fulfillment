from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column , relationship
from datetime import datetime, timezone

from app.database import Base

class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(Integer,primary_key=True,autoincrement=True,index=True,)   
    user_id:Mapped[int] = mapped_column(Integer,index=True)
    amount:Mapped[float] = mapped_column(Float,nullable=False)
    currency:Mapped[str] = mapped_column(String(10), default='UZS')
    country: Mapped[str] = mapped_column(String(100),nullable=False)
    city: Mapped[str] = mapped_column(String(100),nullable=False)
    transactions_last_10_min: Mapped[int] = mapped_column(Integer,default=0)
    is_new_device: Mapped[bool] = mapped_column(Boolean,default=False)
    created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    nullable=False)
    fraud_result: Mapped["FraudResult | None"] = relationship(back_populates="transaction",uselist=False)

class FraudResult(Base):
    __tablename__ = "fraud_results"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id"),
        nullable=False,
        unique=True
    )

    risk_score: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False
    )

    reasons: Mapped[str] = mapped_column(
        String,
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow
    )

    transaction: Mapped["Transaction"] = relationship(
        back_populates="fraud_result"
    )


class User(Base):
    __tablename__ = "users"
    id:Mapped[int] = mapped_column(Integer, primary_key = True, index = True)
    first_name:Mapped[str] = mapped_column(String(100),nullable=False)
    last_name:Mapped[str] = mapped_column(String(100),nullable=False)
    balance:Mapped[float] = mapped_column(Float, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    nullable=False)