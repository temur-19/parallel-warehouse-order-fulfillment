from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import create_tables, get_db
from app.fraud.notification_bot import notify_transaction
from app.ml.predict import predict_transaction
from app.models import FraudResult, Transaction, User
from app.schemas import LoginRequest, TransactionCreate, UserCreate
from app.security import (
    create_access_token,
    get_current_user,
    hash_password,
    is_password_hash,
    verify_password,
)


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


@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/login/", include_in_schema=False)
async def login_page():
    return FileResponse(Path(__file__).parent / "static" / "login.html")


@app.get("/register/", include_in_schema=False)
async def register_page():
    return FileResponse(Path(__file__).parent / "static" / "register.html")


@transaction_router.post("/add/")
async def create_transaction(
    transaction_in: TransactionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sender = await db.execute(select(User).where(User.card_number == transaction_in.sender_card_number))
    sender_obj = sender.scalar_one_or_none()
    if not sender_obj:
        raise HTTPException(status_code=404, detail="Sender not found")
    receiver = await db.execute(select(User).where(User.card_number == transaction_in.receiver_card_number))
    receiver_obj = receiver.scalar_one_or_none()
    if not receiver_obj:
        raise HTTPException(status_code=404, detail='Qabul qiluvchi topilmadi')
    if sender_obj.id != current_user.id:
        raise HTTPException(status_code=403, detail="Faqat o‘z hisobingizdan pul yuborishingiz mumkin")
    if transaction_in.sender_card_number == transaction_in.receiver_card_number:
        raise HTTPException(status_code=400, detail="O'zingizga pul o'tkaza olmaysiz")
    old_balance = sender_obj.balance
    oldbalance_dest = receiver_obj.balance
    sender_id = sender_obj.id
    receiver_id = receiver_obj.id
    if old_balance < transaction_in.amount:
        raise HTTPException(status_code=400, detail="Tranzaksiya uchun balans yetarli emas")

    prediction_in = {
                     'amount': transaction_in.amount,
                     'currency': transaction_in.currency,
                     'country': transaction_in.country,
                     'city': transaction_in.city,
                     'transactions_last_10_min': transaction_in.transactions_last_10_min,
                     'is_new_device': transaction_in.is_new_device,
                     'step': transaction_in.step,
                     'type': transaction_in.type,
                     'oldbalanceOrg': old_balance,
                     'oldbalanceDest': oldbalance_dest,
                     'isFlaggedFraud': transaction_in.is_flagged_fraud  
                     }


    prediction = await predict_transaction(prediction_in)
    status = "fraud" if prediction["is_fraud"] else "legitimate"
    fraud_result = FraudResult(
        transaction_id=prediction_data.id,
        risk_score=prediction["risk_score"],
        status=status,
        reasons=json.dumps(prediction["reasons"], ensure_ascii=False),
    )
    
    db.add(fraud_result)

    await db.commit()
    print("aajhgasjghda", prediction)
    if prediction['risk_score']>0.90:
        try:
                await notify_transaction(
                    {
                    "id": prediction_data.id,
                    "sender_card_number": transaction_in.sender_card_number,
                    "receiver_card_number": transaction_in.receiver_card_number,
                    "amount": transaction_in.amount,
                    "currency": transaction_in.currency,
                    "city": transaction_in.city,
                    "risk_score": float(prediction["risk_score"]),
                    "is_fraud": bool(prediction["is_fraud"]),
                    "status": status,
                     }
                     )    
        except Exception:
                # Xatoni logga yozish kerak.
                # Tranzaksiyani qayta ishlash siyosati
                # loyihaning talablariga bog'liq.
                pass

    elif 0.90 > prediction["risk_score"]>=0.60:
        pass
    else:
        prediction_data = Transaction(
                sender_id = sender_id,
                receiver_id = receiver_id,
                sender_card_number = transaction_in.sender_card_number,
                receiver_card_number = transaction_in.receiver_card_number,
                amount=transaction_in.amount,
                currency=transaction_in.currency,
                country=transaction_in.country,
                city=transaction_in.city,
                transactions_last_10_min=transaction_in.transactions_last_10_min,
                is_new_device=transaction_in.is_new_device,
            )
        new_balance = old_balance - transaction_in.amount
        sender_obj.balance = new_balance

        db.add(prediction_data)
        await db.flush()


    return {
        "transaction_id": prediction_data.id,
        "risk_score": prediction["risk_score"],
        "is_fraud": prediction["is_fraud"],
        "reasons": prediction["reasons"],
        "status": status,
    }


@transaction_router.get("/get/{transaction_id}/")
async def get_transaction(
    transaction_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    transaction = await db.get(Transaction, transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Tranzaksiya topilmadi")
    if current_user.id not in (transaction.sender_id, transaction.receiver_id):
        raise HTTPException(status_code=403, detail="Bu tranzaksiyani ko‘rishga ruxsatingiz yo‘q")
    result = await db.execute(select(FraudResult).where(FraudResult.transaction_id == transaction_id))
    fraud_result = result.scalar_one_or_none()
    if not fraud_result:
        raise HTTPException(
            status_code=404,
            detail="Bu tranzaksiya uchun firibgarlik natijasi topilmadi",
        )

    return {
        "transaction_id": transaction.id,
        "sender": transaction.sender_card_number,
        "receiver": transaction.receiver_card_number,
        "amount": transaction.amount,
        "currency": transaction.currency,
        "country": transaction.country,
        "city": transaction.city,
        "transactions_last_10_min": transaction.transactions_last_10_min,
        "is_new_device": transaction.is_new_device,
        "created_at": transaction.created_at.isoformat(),
        "risk_score": fraud_result.risk_score,
        "status": fraud_result.status,
        "reasons": (
            json.loads(fraud_result.reasons)
            if fraud_result.reasons and fraud_result.reasons.startswith("[")
            else fraud_result.reasons.splitlines()
            if fraud_result.reasons
            else []
        ),
    }

@user_router.post('/add/')
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    user = User(
        phone = user_in.phone,
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        password=hash_password(user_in.password),
        balance=user_in.balance,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {
        'user.id':user.id,
        'user.name':user.first_name,
        'user.last_name':user.last_name,
        'user.balance':user.balance,
        'card_number':user.card_number,
        'created_at':user.created_at    
    }

@user_router.post("/login/")
async def login(login_in: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(
        select(User).where(User.phone == login_in.phone)
    )
    if not user:
        raise HTTPException(status_code=401, detail="Karta raqami yoki parol noto‘g‘ri")
    if not verify_password(login_in.password, user.password):
        raise HTTPException(status_code=401, detail="Karta raqami yoki parol noto‘g‘ri")

    if not is_password_hash(user.password):
        user.password = hash_password(login_in.password)
        await db.commit()

    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "user_id": user.id,
        "card_number": user.card_number,
        "first_name": user.first_name,
        "last_name": user.last_name,
    }
