from contextlib import asynccontextmanager
from decimal import Decimal
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import create_tables, get_db
from app.fraud.notification_bot import notify_transaction
from app.ml.predict import predict_transaction
from app.models import FraudResult, Transaction, User, Cards
from app.schemas import (
    GetProfil,
    GetUserBalance,
    GetUserCards,
    LoginRequest,
    TransactionCreate,
    UserCreate,
)
from app.security import (
    create_access_token,
    get_current_user,
    hash_password,
    is_password_hash,
    verify_password,
)

logger = logging.getLogger(__name__)


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
    sender_stmt = select(Cards).where(
        Cards.user_id == current_user.id,
        Cards.is_active.is_(True),
        Cards.id == transaction_in.sender_card_id,
    )

    sender = await db.execute(sender_stmt.with_for_update())
    sender_obj = sender.scalar_one_or_none()
    if not sender_obj:
        raise HTTPException(status_code=404, detail="Karta topilmadi")
    receiver = await db.execute(
        select(Cards).where(
            transaction_in.receiver_card_number == Cards.pan_masked,
            Cards.is_active.is_(True),
        )
    )
    receiver_obj = receiver.scalar_one_or_none()
    if not receiver_obj:
        raise HTTPException(status_code=404, detail='Qabul qiluvchi topilmadi')

    if sender_obj.user_id == receiver_obj.user_id:
        raise HTTPException(status_code=400, detail="O'zingizga pul o'tkaza olmaysiz")
    old_balance = sender_obj.balance
    oldbalance_dest = receiver_obj.balance
    sender_id = sender_obj.user_id
    receiver_id = receiver_obj.user_id
    transfer_amount = Decimal(str(transaction_in.amount))
    if old_balance < transfer_amount:
        raise HTTPException(status_code=400, detail="Tranzaksiya uchun balans yetarli emas")

    prediction_in = {
        "amount": transaction_in.amount,
        "step": transaction_in.step,
        "type": transaction_in.type,
        "oldbalanceOrg": old_balance,
        "oldbalanceDest": oldbalance_dest,
        "isFlaggedFraud": transaction_in.is_flagged_fraud,
    }

    prediction = await predict_transaction(prediction_in)
    status = "fraud" if prediction["is_fraud"] else "legitimate"
    risk_score = float(prediction["risk_score"])
    if prediction["is_fraud"]:
        transaction_status = "Pending Review"
        risk_level = "HIGH"
    elif risk_score >= 0.60:
        transaction_status = "Pending Review"
        risk_level = "Medium"
    else:
        transaction_status = "COMPLETED"
        risk_level = "LOW"

    prediction_data = Transaction(
        sender_id=sender_id,
        receiver_id=receiver_id,
        sender_card_number=sender_obj.id,
        receiver_card_number=receiver_obj.id,
        amount=transaction_in.amount,
        currency=transaction_in.currency,
        country=transaction_in.country,
        city=transaction_in.city,
        transactions_last_10_min=transaction_in.transactions_last_10_min,
        is_new_device=transaction_in.is_new_device,
        transaction_status=transaction_status,
    )
    db.add(prediction_data)
    await db.flush()

    fraud_result = FraudResult(
        transaction_id=prediction_data.id,
        risk_score=risk_score,
        status=status,
        reasons=json.dumps(prediction["reasons"], ensure_ascii=False),
    )
    db.add(fraud_result)

    if transaction_status == "COMPLETED":
        sender_obj.balance = old_balance - transfer_amount
        receiver_obj.balance = oldbalance_dest + transfer_amount

    await db.commit()

    if transaction_status == "Pending Review":
        try:
            await notify_transaction(
                {
                    "id": prediction_data.id,
                    "sender_card_number": f"•••• {sender_obj.pan_masked[-4:]}",
                    "receiver_card_number": f"•••• {receiver_obj.pan_masked[-4:]}",
                    "amount": transaction_in.amount,
                    "currency": transaction_in.currency,
                    "city": transaction_in.city,
                    "risk_score": risk_score,
                    "is_fraud": bool(prediction["is_fraud"]),
                    "status": status,
                    "transaction_status": transaction_status,
                    "reasons": prediction["reasons"],
                }
            )
        except (RuntimeError, httpx.HTTPError):
            logger.exception(
                "Could not send notification for transaction %s",
                prediction_data.id,
            )

    return {
        "transaction_id": prediction_data.id,
        "transaction_status": transaction_status,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "is_fraud": prediction["is_fraud"],
        "reasons": prediction["reasons"],
        "status": status,
    }


@user_router.get("/me/", response_model=GetProfil)
@user_router.get("/profile/", response_model=GetProfil)
@user_router.get("/get/profil", response_model=GetProfil)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    return {
        "user_id": current_user.id,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "phone": current_user.phone,
    }


@user_router.get("/cards/", response_model=list[GetUserCards])
async def get_user_cards(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Cards)
        .where(Cards.user_id == current_user.id, Cards.is_active.is_(True))
        .order_by(Cards.is_default.desc(), Cards.id)
    )
    cards = result.scalars().all()
    return [
        {
            "id": card.id,
            "last4": card.pan_masked[-4:],
            "display_pan": f"••••  ••••  ••••  {card.pan_masked[-4:]}",
            "card_number": card.pan_masked,
            "cardholder_name": f"{current_user.first_name} {current_user.last_name}",
            "balance": float(card.balance or 0),
            "currency": "UZS",
            "is_default": card.is_default,
        }
        for card in cards
    ]


@user_router.get("/balance/", response_model=GetUserBalance)
async def get_user_balance(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Cards.balance).where(
            Cards.user_id == current_user.id,
            Cards.is_active.is_(True),
        )
    )
    balances = result.scalars().all()
    total_balance = sum(balances, Decimal("0.00"))
    return {
        "total_balance": total_balance,
        "currency": "UZS",
        "cards_count": len(balances),
    }


@transaction_router.get("/history/")
async def get_transaction_history(
    limit: int = Query(default=10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Transaction, FraudResult, Cards)
        .outerjoin(
            FraudResult, FraudResult.transaction_id == Transaction.id
        )
        .join(Cards, Cards.id == Transaction.sender_card_number)
        .where(
            (Transaction.sender_id == current_user.id)
            | (Transaction.receiver_id == current_user.id)
        )
        .order_by(Transaction.created_at.desc())
        .limit(limit)
    )
    transactions = result.all()
    card_ids = {
        transaction.receiver_card_number for transaction, _, _ in transactions
    }
    cards_result = await db.execute(select(Cards).where(Cards.id.in_(card_ids)))
    cards_by_id = {card.id: card for card in cards_result.scalars().all()}

    return [
        {
            "transaction_id": transaction.id,
            "sender": f"•••• {sender_card.pan_masked[-4:]}",
            "receiver": (
                f"•••• {cards_by_id[transaction.receiver_card_number].pan_masked[-4:]}"
                if transaction.receiver_card_number in cards_by_id
                else "••••"
            ),
            "amount": transaction.amount,
            "currency": transaction.currency,
            "city": transaction.city,
            "created_at": transaction.created_at.isoformat(),
            "transaction_status": transaction.transaction_status,
            "status": fraud_result.status if fraud_result else "unknown",
            "risk_score": fraud_result.risk_score if fraud_result else None,
            "direction": (
                "outgoing"
                if transaction.sender_id == current_user.id
                else "incoming"
            ),
        }
        for transaction, fraud_result, sender_card in transactions
    ]




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

    sender_card = await db.get(Cards, transaction.sender_card_number)
    receiver_card = await db.get(Cards, transaction.receiver_card_number)
    return {
        "transaction_id": transaction.id,
        "sender": f"•••• {sender_card.pan_masked[-4:]}" if sender_card else None,
        "receiver": f"•••• {receiver_card.pan_masked[-4:]}" if receiver_card else None,
        "amount": transaction.amount,
        "currency": transaction.currency,
        "country": transaction.country,
        "city": transaction.city,
        "transactions_last_10_min": transaction.transactions_last_10_min,
        "is_new_device": transaction.is_new_device,
        "created_at": transaction.created_at.isoformat(),
        "transaction_status": transaction.transaction_status,
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
    existing_user = await db.scalar(
        select(User).where(User.phone == user_in.phone)
    )
    if existing_user:
        raise HTTPException(
            status_code=409, detail="Bu telefon raqam ro‘yxatdan o‘tgan"
        )

    user = User(
        phone = user_in.phone,
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        password=hash_password(user_in.password),
    )
    db.add(user)
    await db.flush()
    card = Cards(
        user_id=user.id,
        balance=Decimal(str(user_in.balance)),
        is_default=True,
        is_active=True,
    )
    db.add(card)
    await db.commit()
    await db.refresh(card)
    return {
        'user_id':user.id,
        'first_name':user.first_name,
        'last_name':user.last_name,
        'card_id':card.id,
        'card_number':card.pan_masked,
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
        "first_name": user.first_name,
        "last_name": user.last_name,
    }