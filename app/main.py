from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import create_tables, get_db
from app.ml.predict import predict_transaction
from app.models import FraudResult, Transaction, User
from app.schemas import TransactionCreate, UserCreate


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
user_router = APIRouter(prefix="/user")


app.include_router(transaction_router)
app.include_router(user_router)


@transaction_router.post("/add/")
async def create_transaction(
    transaction_in: TransactionCreate,
    db: AsyncSession = Depends(get_db)
):
    # user = await db.get(User, transaction_in.user_id)
    # if not user:
    #     return {"error": "User not found"}
    
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

    prediction = await predict_transaction(transaction_in.model_dump(by_alias=True))
    status = "fraud" if prediction["is_fraud"] else "legitimate"
    fraud_result = FraudResult(
        transaction_id=prediction_data.id,
        risk_score=prediction["risk_score"],
        status=status,
    )

    db.add(fraud_result)

    await db.commit()
    print("aajhgasjghda", prediction)
    return {
        "transaction_id": prediction_data.id,
        "risk_score": prediction["risk_score"],
        "is_fraud": prediction["is_fraud"],
        "status": status,
    }


@transaction_router.get('get/{transaction_id}/')
async def get_transaction(transaction_id: int, db: AsyncSession = Depends(get_db)):
    transaction = await db.get(Transaction, transaction_id)
    if not transaction:
        return {"error": "Transaction not found"}

    fraud_result = await db.get(FraudResult, transaction_id)
    if not fraud_result:
        return {"error": "Fraud result not found for this transaction"}

    return {
        "transaction_id": transaction.id,
        "user_id": transaction.user_id,
        "amount": transaction.amount,
        "currency": transaction.currency,
        "country": transaction.country,
        "city": transaction.city,
        "transactions_last_10_min": transaction.transactions_last_10_min,
        "is_new_device": transaction.is_new_device,
        "created_at": transaction.created_at.isoformat(),
        "risk_score": fraud_result.risk_score,
        "status": fraud_result.status,
        "reasons": fraud_result.reasons,
    }

@user_router.post('/add/')
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    user = User(
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        balance=user_in.balance
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user