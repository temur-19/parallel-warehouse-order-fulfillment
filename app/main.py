from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import create_tables, get_db
from app.ml.predict import predict_transaction
from app.models import FraudResult, Transaction
from app.schemas import TransactionCreate


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Creating database tables...")
    await create_tables()
    print("Database tables created!")
    yield


app = FastAPI(
    title="Parallel Fraud Detection API",
    lifespan=lifespan,
)


transaction_router = APIRouter(prefix="/transaction")


app.include_router(transaction_router)


@transaction_router.post("/add/")
async def create_transaction(
    transaction_in: TransactionCreate,
    db: AsyncSession = Depends(get_db)
):
    prediction_data = Transaction(
        amount=transaction_in.amount,
        user_id=transaction_in.user_id,
        currency=transaction_in.currency,
        country=transaction_in.country,
        city=transaction_in.city,
        transactions_last_10_min=transaction_in.transactions_last_10_min,
        is_new_device=transaction_in.is_new_device,
    )

    db.add(prediction_data)

    await db.flush()

    prediction = predict_transaction(transaction_in.model_dump(by_alias=True))
    status = "fraud" if prediction["is_fraud"] else "legitimate"
    fraud_result = FraudResult(
        transaction_id=prediction_data.id,
        risk_score=prediction["risk_score"],
        status=status,
    )

    db.add(fraud_result)

    await db.commit()

    return {
        "transaction_id": prediction_data.id,
        "risk_score": prediction["risk_score"],
        "is_fraud": prediction["is_fraud"],
        "status": status,
    }