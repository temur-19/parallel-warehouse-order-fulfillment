from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Boolean,
    Numeric,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column , relationship
from datetime import datetime, timezone
import random
from decimal import Decimal

from app.database import Base

def generate_card_number():
    num = '9860'
    return num + ''.join(str(random.randint(1000, 9999)) for i in range(3))

class User(Base):
    __tablename__ = "users"
    id:Mapped[int] = mapped_column(Integer, primary_key = True, index = True)
    phone:Mapped[str] = mapped_column(String(13), nullable=False)
    first_name:Mapped[str] = mapped_column(String(100),nullable=False)
    last_name:Mapped[str] = mapped_column(String(100),nullable=False)
    password:Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc),
    nullable=False    )


class Cards(Base):
    __tablename__ = 'cards'
    id:Mapped[int] = mapped_column(Integer, primary_key=True, index = True)
    user_id:Mapped[int] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    pan_masked: Mapped[str] = mapped_column(String(16), nullable=False, default=generate_card_number, unique=True)
    balance:Mapped[Decimal] = mapped_column(Numeric(18, 2), default = 0)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(Integer,primary_key=True,autoincrement=True,index=True)   
    sender_id:Mapped[int] = mapped_column(ForeignKey("users.id"),nullable=False,index=True)
    receiver_id:Mapped[int] = mapped_column(ForeignKey("users.id"),nullable=False,index=True)
    amount:Mapped[float] = mapped_column(Float,nullable=False)
    sender_card_number: Mapped[int] = mapped_column(
        ForeignKey("cards.id"), nullable=False
    )
    receiver_card_number: Mapped[int] = mapped_column(
        ForeignKey("cards.id"), nullable=False
    )
    currency:Mapped[str] = mapped_column(String(10), default='UZS')
    country: Mapped[str] = mapped_column(String(100),nullable=False)
    city: Mapped[str] = mapped_column(String(100),nullable=False)
    transactions_last_10_min: Mapped[int] = mapped_column(Integer,default=0)
    is_new_device: Mapped[bool] = mapped_column(Boolean,default=False)
    transaction_status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="COMPLETED"
    )
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
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    transaction: Mapped["Transaction"] = relationship(
        back_populates="fraud_result"
    )




class Fraud_cases(Base):
    __tablename__ = 'fraud_cases'
    id:Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id:Mapped[int] = mapped_column(ForeignKey('transactions.id'), nullable=False, index=True)
    risk_score:Mapped[str] = mapped_column(String)
    status:Mapped[str] = mapped_column(String)


class UserDevice(Base):
    __tablename__ = "user_devices"
    __table_args__ = (
        UniqueConstraint("user_id", "device_hash", name="uq_user_device_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )
    device_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    device_name: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class TransactionReview(Base):
    __tablename__ = "transaction_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id"), nullable=False, unique=True, index=True
    )
    admin_telegram_id: Mapped[str] = mapped_column(String(32), nullable=False)
    decision: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
