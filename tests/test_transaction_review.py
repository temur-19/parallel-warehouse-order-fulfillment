import unittest
from unittest.mock import patch

import httpx
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.main as main
import app.fraud.notification_bot as notification_bot
from app.database import Base, get_db
from app.models import Cards, FraudResult, Transaction, TransactionReview, User
from app.schemas import TransactionCreate


class TransactionReviewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        self.session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

        async with self.session_factory() as session:
            sender = User(
                phone="+998901234567",
                first_name="Ali",
                last_name="Valiyev",
                password="hash",
            )
            recipient = User(
                phone="+998909876543",
                first_name="Vali",
                last_name="Karimov",
                password="hash",
            )
            session.add_all((sender, recipient))
            await session.flush()
            self.sender_card = Cards(
                user_id=sender.id,
                pan_masked="9860000000000001",
                balance=100_000,
                is_active=True,
            )
            self.recipient_card = Cards(
                user_id=recipient.id,
                pan_masked="9860000000000002",
                balance=1_000,
                is_active=True,
            )
            session.add_all((self.sender_card, self.recipient_card))
            await session.flush()
            self.transaction = Transaction(
                sender_id=sender.id,
                receiver_id=recipient.id,
                amount=10_000,
                sender_card_number=self.sender_card.id,
                receiver_card_number=self.recipient_card.id,
                currency="UZS",
                country="Uzbekistan",
                city="Tashkent",
                transaction_status="Pending Review",
            )
            self.insufficient_transaction = Transaction(
                sender_id=sender.id,
                receiver_id=recipient.id,
                amount=95_000,
                sender_card_number=self.sender_card.id,
                receiver_card_number=self.recipient_card.id,
                currency="UZS",
                country="Uzbekistan",
                city="Tashkent",
                transaction_status="Pending Review",
            )
            session.add_all((self.transaction, self.insufficient_transaction))
            await session.flush()
            self.transaction_id = self.transaction.id
            self.insufficient_transaction_id = self.insufficient_transaction.id
            session.add_all(
                (
                    FraudResult(
                        transaction_id=self.transaction_id,
                        risk_score=0.9,
                        status="fraud",
                        reasons="[]",
                    ),
                    FraudResult(
                        transaction_id=self.insufficient_transaction_id,
                        risk_score=0.9,
                        status="fraud",
                        reasons="[]",
                    ),
                )
            )
            await session.commit()

        async def override_db():
            async with self.session_factory() as session:
                yield session

        self.original_admin_id = main.ADMIN_ID
        self.original_webhook_secret = main.WEBHOOK_SECRET
        self.original_answer = main.answer_callback_query
        self.original_edit = main.edit_telegram_message
        self.original_db_override = main.app.dependency_overrides.get(get_db)
        main.ADMIN_ID = "123456"
        main.WEBHOOK_SECRET = "a" * 40
        main.app.dependency_overrides[get_db] = override_db
        self.callback_responses = []

        async def answer_callback(callback_id, text, show_alert=False):
            self.callback_responses.append((callback_id, text, show_alert))

        async def edit_message(chat_id, message_id, text):
            self.callback_responses.append((chat_id, message_id, text))

        main.answer_callback_query = answer_callback
        main.edit_telegram_message = edit_message
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        main.ADMIN_ID = self.original_admin_id
        main.WEBHOOK_SECRET = self.original_webhook_secret
        main.answer_callback_query = self.original_answer
        main.edit_telegram_message = self.original_edit
        if self.original_db_override is None:
            main.app.dependency_overrides.pop(get_db, None)
        else:
            main.app.dependency_overrides[get_db] = self.original_db_override
        await self.engine.dispose()

    def _callback_payload(self, transaction_id, callback_id):
        return {
            "callback_query": {
                "id": callback_id,
                "from": {"id": 123456},
                "message": {
                    "chat": {"id": 123456},
                    "message_id": 7,
                },
                "data": f"approve:{transaction_id}",
            }
        }

    async def _approve(self, transaction_id, callback_id):
        return await self.client.post(
            "/telegram/webhook",
            headers={
                "X-Telegram-Bot-Api-Secret-Token": main.WEBHOOK_SECRET
            },
            json=self._callback_payload(transaction_id, callback_id),
        )

    async def test_approval_is_once_and_never_overdraws(self):
        first = await self._approve(self.transaction_id, "approve-1")
        replay = await self._approve(self.transaction_id, "approve-2")
        insufficient = await self._approve(
            self.insufficient_transaction_id, "approve-3"
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(insufficient.status_code, 200)
        async with self.session_factory() as session:
            sender_balance = await session.scalar(
                select(Cards.balance).where(Cards.id == self.sender_card.id)
            )
            recipient_balance = await session.scalar(
                select(Cards.balance).where(
                    Cards.id == self.recipient_card.id
                )
            )
            approved = await session.get(Transaction, self.transaction_id)
            still_pending = await session.get(
                Transaction, self.insufficient_transaction_id
            )
            reviews = list(
                (await session.scalars(select(TransactionReview))).all()
            )

        self.assertEqual(sender_balance, 90_000)
        self.assertEqual(recipient_balance, 11_000)
        self.assertEqual(approved.transaction_status, "COMPLETED")
        self.assertEqual(still_pending.transaction_status, "Pending Review")
        self.assertEqual(len(reviews), 1)

    async def test_webhook_secret_and_admin_identity_are_required(self):
        bad_secret = await self.client.post(
            "/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
            json=self._callback_payload(self.transaction_id, "bad-secret"),
        )
        non_admin_payload = self._callback_payload(
            self.transaction_id, "non-admin"
        )
        non_admin_payload["callback_query"]["from"]["id"] = 987654
        non_admin = await self.client.post(
            "/telegram/webhook",
            headers={
                "X-Telegram-Bot-Api-Secret-Token": main.WEBHOOK_SECRET
            },
            json=non_admin_payload,
        )

        self.assertEqual(bad_secret.status_code, 403)
        self.assertEqual(non_admin.status_code, 200)
        async with self.session_factory() as session:
            sender_balance = await session.scalar(
                select(Cards.balance).where(Cards.id == self.sender_card.id)
            )
            transaction = await session.get(Transaction, self.transaction_id)
        self.assertEqual(sender_balance, 100_000)
        self.assertEqual(transaction.transaction_status, "Pending Review")

    def test_clients_cannot_supply_server_derived_risk_signals(self):
        with self.assertRaises(ValidationError):
            TransactionCreate.model_validate(
                {
                    "sender_card_id": 1,
                    "receiver_card_number": "9860000000000002",
                    "amount": 1_000,
                    "country": "Uzbekistan",
                    "city": "Tashkent",
                    "transactions_last_10_min": 1000,
                    "is_new_device": False,
                    "step": 1,
                    "type": "CASH_OUT",
                    "isFlaggedFraud": 1,
                }
            )


class FraudNotificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_alert_shows_model_and_context_decisions(self):
        sent = {}

        async def capture_message(text, reply_markup=None):
            sent["text"] = text
            sent["reply_markup"] = reply_markup
            return True

        with (
            patch.object(
                notification_bot, "send_telegram_message", capture_message
            ),
            patch.object(
                notification_bot, "telegram_approval_enabled", return_value=True
            ),
        ):
            await notification_bot.notify_transaction(
                {
                    "id": 13,
                    "transaction_status": "Pending Review",
                    "amount": 125_000,
                    "currency": "UZS",
                    "sender_name": "Ali Valiyev",
                    "sender_phone": "+998901234567",
                    "sender_card_number": "•••• 0001",
                    "recipient_name": "Vali K.",
                    "receiver_card_number": "•••• 0002",
                    "city": "Tashkent",
                    "country": "Uzbekistan",
                    "device_name": "Test Browser",
                    "is_new_device": True,
                    "transactions_last_10_min": 4,
                    "risk_score": 0.6,
                    "model_risk_score": 0.12,
                    "contextual_risk_score": 0.6,
                    "fraud_threshold": 0.85,
                    "review_threshold": 0.5,
                    "is_fraud": False,
                    "model_inputs": {
                        "type": "TRANSFER",
                        "step": 25,
                        "amount": 125_000,
                        "oldbalanceOrg": 250_000,
                        "oldbalanceDest": 50_000,
                        "isFlaggedFraud": 0,
                    },
                    "reasons": [
                        {
                            "feature": "Yangi qurilma",
                            "description": "Qurilma bazada yo‘q",
                        },
                    ],
                }
            )

        self.assertIn("Model ehtimoli: 12.0%", sent["text"])
        self.assertIn("Model fraud chegarasi: 85.0%", sent["text"])
        self.assertIn("Kontekst xavfi (qurilma/tezlik): 60.0%", sent["text"])
        self.assertIn("Yakuniy risk: 60.0%", sent["text"])
        self.assertIn("Yakuniy qaror: Pending Review (MEDIUM)", sent["text"])
        self.assertIn("Jo‘natuvchi oldingi balansi: 250,000.00", sent["text"])
        self.assertIn("Risk signallari:", sent["text"])
        self.assertEqual(
            sent["reply_markup"]["inline_keyboard"][1][0]["callback_data"],
            "approve:13",
        )


if __name__ == "__main__":
    unittest.main()
