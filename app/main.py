from contextlib import asynccontextmanager
from decimal import Decimal
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import logging
from pathlib import Path
import re

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, update

from app.database import create_tables, get_db
from app.fraud.bot import (
    ADMIN_ID,
    WEBHOOK_SECRET,
    answer_callback_query,
    configure_telegram_webhook,
    edit_telegram_message,
)
from app.fraud.notification_bot import notify_transaction
from app.ml.predict import REVIEW_RISK_THRESHOLD, predict_transaction
from app.models import (
    Cards,
    FraudResult,
    Transaction,
    TransactionReview,
    User,
    UserDevice,
)
from app.schemas import (
    GetProfil,
    GetUserBalance,
    GetUserCards,
    LoginRequest,
    RecipientLookup,
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
    try:
        await configure_telegram_webhook()
    except (RuntimeError, httpx.HTTPError):
        logger.exception("Could not configure the Telegram webhook")
        raise
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
    request: Request,
    device_id: str | None = Header(default=None, alias="X-Device-ID"),
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

    now = datetime.now(timezone.utc)
    recent_transaction_count = await db.scalar(
        select(func.count(Transaction.id)).where(
            Transaction.sender_id == sender_id,
            Transaction.created_at >= now - timedelta(minutes=10),
        )
    )
    recent_transaction_count = int(recent_transaction_count or 0)

    user_agent = request.headers.get("user-agent", "")[:512]
    is_new_device = True
    device_name = user_agent or "Noma’lum qurilma"
    if device_id:
        if not 16 <= len(device_id) <= 128:
            raise HTTPException(
                status_code=400,
                detail="Qurilma identifikatori noto‘g‘ri",
            )
        device_hash = hashlib.sha256(
            f"{current_user.id}:{device_id}".encode("utf-8")
        ).hexdigest()
        device = await db.scalar(
            select(UserDevice).where(
                UserDevice.user_id == current_user.id,
                UserDevice.device_hash == device_hash,
            )
        )
        is_new_device = device is None
        if device is None:
            db.add(
                UserDevice(
                    user_id=current_user.id,
                    device_hash=device_hash,
                    device_name=user_agent,
                    first_seen_at=now,
                    last_seen_at=now,
                )
            )
        else:
            device.last_seen_at = now

    prediction_in = {
        "amount": transaction_in.amount,
        "step": (int(now.timestamp() // 3600) % 744) + 1,
        "type": "TRANSFER",
        "oldbalanceOrg": old_balance,
        "oldbalanceDest": oldbalance_dest,
        "isFlaggedFraud": 0,
        "transactions_last_10_min": recent_transaction_count,
        "is_new_device": is_new_device,
    }

    prediction = await predict_transaction(prediction_in)
    status = "fraud" if prediction["is_fraud"] else "legitimate"
    risk_score = float(prediction["risk_score"])
    if prediction["is_fraud"]:
        transaction_status = "Pending Review"
        risk_level = "HIGH"
    elif risk_score >= REVIEW_RISK_THRESHOLD:
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
        transactions_last_10_min=recent_transaction_count,
        is_new_device=is_new_device,
        transaction_status=transaction_status,
        created_at=now,
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
        debit = await db.execute(
            update(Cards)
            .where(
                Cards.id == sender_obj.id,
                Cards.user_id == sender_id,
                Cards.is_active.is_(True),
                Cards.balance >= transfer_amount,
            )
            .values(balance=Cards.balance - transfer_amount)
        )
        if debit.rowcount != 1:
            await db.rollback()
            raise HTTPException(
                status_code=400,
                detail="Tranzaksiya uchun balans yetarli emas",
            )
        credit = await db.execute(
            update(Cards)
            .where(
                Cards.id == receiver_obj.id,
                Cards.user_id == receiver_id,
                Cards.is_active.is_(True),
            )
            .values(balance=Cards.balance + transfer_amount)
        )
        if credit.rowcount != 1:
            await db.rollback()
            raise HTTPException(
                status_code=409,
                detail="Qabul qiluvchi karta faol emas",
            )

    await db.commit()

    admin_notified: bool | None = None
    if transaction_status == "Pending Review":
        try:
            sender_user = await db.get(User, sender_id)
            recipient_user = await db.get(User, receiver_id)
            if sender_user is None or recipient_user is None:
                raise RuntimeError("Tranzaksiya foydalanuvchisi topilmadi")
            await notify_transaction(
                {
                    "id": prediction_data.id,
                    "sender_card_number": f"•••• {sender_obj.pan_masked[-4:]}",
                    "receiver_card_number": f"•••• {receiver_obj.pan_masked[-4:]}",
                    "sender_name": (
                        f"{sender_user.first_name} {sender_user.last_name}"
                    ),
                    "sender_phone": sender_user.phone,
                    "recipient_name": (
                        f"{recipient_user.first_name} "
                        f"{recipient_user.last_name[:1].upper() or '?'}."
                    ),
                    "amount": transaction_in.amount,
                    "currency": transaction_in.currency,
                    "city": transaction_in.city,
                    "country": transaction_in.country,
                    "device_name": device_name,
                    "is_new_device": is_new_device,
                    "transactions_last_10_min": recent_transaction_count,
                    "risk_score": risk_score,
                    "model_risk_score": float(prediction["model_risk_score"]),
                    "contextual_risk_score": float(
                        prediction["contextual_risk_score"]
                    ),
                    "fraud_threshold": float(prediction["fraud_threshold"]),
                    "review_threshold": REVIEW_RISK_THRESHOLD,
                    "is_fraud": bool(prediction["is_fraud"]),
                    "status": status,
                    "transaction_status": transaction_status,
                    "reasons": prediction["reasons"],
                    "model_inputs": prediction_in,
                }
            )
            admin_notified = True
        except (RuntimeError, httpx.HTTPError):
            admin_notified = False
            logger.exception(
                "Could not send notification for transaction %s",
                prediction_data.id,
            )

    return {
        "transaction_id": prediction_data.id,
        "transaction_status": transaction_status,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "model_risk_score": float(prediction["model_risk_score"]),
        "contextual_risk_score": float(prediction["contextual_risk_score"]),
        "fraud_threshold": float(prediction["fraud_threshold"]),
        "review_threshold": REVIEW_RISK_THRESHOLD,
        "is_fraud": prediction["is_fraud"],
        "reasons": prediction["reasons"],
        "status": status,
        "admin_notified": admin_notified,
    }


@transaction_router.post("/recipient/lookup/")
async def lookup_recipient(
    recipient_in: RecipientLookup,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    recipient = await db.scalar(
        select(Cards).where(
            Cards.pan_masked == recipient_in.card_number,
            Cards.is_active.is_(True),
        )
    )
    if recipient is None:
        raise HTTPException(status_code=404, detail="Faol karta topilmadi")
    owner = await db.get(User, recipient.user_id)
    if owner is None:
        raise HTTPException(status_code=404, detail="Karta egasi topilmadi")
    if owner.id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="O‘zingizning kartangizga pul o‘tkaza olmaysiz",
        )
    return {
        "cardholder_name": (
            f"{owner.first_name} {owner.last_name[:1].upper() or '?'}."
        ),
    }


@app.post("/telegram/webhook", include_in_schema=False)
async def telegram_webhook(
    payload: dict,
    db: AsyncSession = Depends(get_db),
    secret_token: str | None = Header(
        default=None, alias="X-Telegram-Bot-Api-Secret-Token"
    ),
):
    if not WEBHOOK_SECRET or not hmac.compare_digest(
        secret_token or "", WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=403, detail="Webhook token noto‘g‘ri")
    if not ADMIN_ID or not ADMIN_ID.isdigit():
        raise HTTPException(
            status_code=503,
            detail="Telegram admin ID sozlanmagan",
        )

    callback = payload.get("callback_query")
    if not isinstance(callback, dict):
        return {"ok": True}
    callback_id = callback.get("id")
    actor = callback.get("from")
    message = callback.get("message")
    if not isinstance(callback_id, str) or not isinstance(actor, dict):
        return {"ok": True}
    if str(actor.get("id")) != ADMIN_ID:
        await answer_callback_query(
            callback_id, "Bu amal uchun ruxsatingiz yo‘q.", show_alert=True
        )
        return {"ok": True}
    if not isinstance(message, dict):
        return {"ok": True}
    chat = message.get("chat")
    if not isinstance(chat, dict) or str(chat.get("id")) != ADMIN_ID:
        await answer_callback_query(
            callback_id, "Faqat adminning shaxsiy chatida ishlaydi.", show_alert=True
        )
        return {"ok": True}

    callback_data = callback.get("data")
    match = (
        re.fullmatch(r"(contact|approve):([1-9][0-9]*)", callback_data)
        if isinstance(callback_data, str)
        else None
    )
    if match is None:
        await answer_callback_query(callback_id, "Tugma ma’lumoti noto‘g‘ri.")
        return {"ok": True}

    action, transaction_id_text = match.groups()
    transaction = await db.get(Transaction, int(transaction_id_text))
    if transaction is None:
        await answer_callback_query(
            callback_id, "Tranzaksiya topilmadi.", show_alert=True
        )
        return {"ok": True}

    if action == "contact":
        sender = await db.get(User, transaction.sender_id)
        if sender is None:
            await answer_callback_query(
                callback_id, "Foydalanuvchi topilmadi.", show_alert=True
            )
            return {"ok": True}
        await answer_callback_query(
            callback_id,
            f"{sender.first_name} {sender.last_name}\nTelefon: {sender.phone}",
            show_alert=True,
        )
        return {"ok": True}

    if transaction.transaction_status != "Pending Review":
        await answer_callback_query(
            callback_id,
            f"Tranzaksiya allaqachon {transaction.transaction_status}.",
            show_alert=True,
        )
        return {"ok": True}

    claim = await db.execute(
        update(Transaction)
        .where(
            Transaction.id == transaction.id,
            Transaction.transaction_status == "Pending Review",
        )
        .values(transaction_status="APPROVING")
    )
    if claim.rowcount != 1:
        await db.rollback()
        await answer_callback_query(
            callback_id, "Tranzaksiya allaqachon ko‘rib chiqilmoqda."
        )
        return {"ok": True}

    transfer_amount = Decimal(str(transaction.amount))
    debit = await db.execute(
        update(Cards)
        .where(
            Cards.id == transaction.sender_card_number,
            Cards.user_id == transaction.sender_id,
            Cards.is_active.is_(True),
            Cards.balance >= transfer_amount,
        )
        .values(balance=Cards.balance - transfer_amount)
    )
    if debit.rowcount != 1:
        await db.rollback()
        await answer_callback_query(
            callback_id,
            "Jo‘natuvchi kartada mablag‘ yetarli emas. O‘tkazma bajarilmadi.",
            show_alert=True,
        )
        return {"ok": True}

    credit = await db.execute(
        update(Cards)
        .where(
            Cards.id == transaction.receiver_card_number,
            Cards.user_id == transaction.receiver_id,
            Cards.is_active.is_(True),
        )
        .values(balance=Cards.balance + transfer_amount)
    )
    if credit.rowcount != 1:
        await db.rollback()
        await answer_callback_query(
            callback_id,
            "Qabul qiluvchi karta faol emas. O‘tkazma bajarilmadi.",
            show_alert=True,
        )
        return {"ok": True}

    transaction.transaction_status = "COMPLETED"
    fraud_result = await db.scalar(
        select(FraudResult).where(
            FraudResult.transaction_id == transaction.id
        )
    )
    if fraud_result is None:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Tranzaksiya uchun fraud natijasi topilmadi",
        )
    fraud_result.status = "approved_after_review"
    db.add(
        TransactionReview(
            transaction_id=transaction.id,
            admin_telegram_id=ADMIN_ID,
            decision="approved",
        )
    )
    await db.commit()
    await answer_callback_query(
        callback_id, "Shaxs tasdiqlandi va o‘tkazma bajarildi."
    )
    await edit_telegram_message(
        int(chat["id"]),
        int(message["message_id"]),
        f"✅ Tranzaksiya #{transaction.id} tekshirildi va bajarildi.",
    )
    return {"ok": True}


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