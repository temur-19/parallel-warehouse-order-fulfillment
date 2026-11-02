import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID")
WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")


def telegram_approval_enabled() -> bool:
    return bool(WEBHOOK_URL and WEBHOOK_SECRET)


async def _telegram_request(method: str, payload: dict) -> dict:
    if not BOT_TOKEN or not ADMIN_ID:
        raise RuntimeError("Telegram konfiguratsiyasi topilmadi")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

        try:
            result = response.json()
        except ValueError as error:
            raise RuntimeError("Telegram API yaroqsiz javob qaytardi") from error

        if not isinstance(result, dict) or not result.get("ok"):
            description = (
                result.get("description", "noma’lum sabab")
                if isinstance(result, dict)
                else "kutilmagan javob formati"
            )
            raise RuntimeError(
                f"Telegram so‘rovi bajarilmadi: {description}"
            )

        return result


async def send_telegram_message(
    text: str, reply_markup: dict | None = None
) -> bool:
    payload: dict[str, object] = {
        "chat_id": ADMIN_ID,
        "text": text,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    await _telegram_request("sendMessage", payload)
    return True


async def answer_callback_query(
    callback_query_id: str, text: str, show_alert: bool = False
) -> None:
    await _telegram_request(
        "answerCallbackQuery",
        {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": show_alert,
        },
    )


async def edit_telegram_message(
    chat_id: int, message_id: int, text: str
) -> None:
    await _telegram_request(
        "editMessageText",
        {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        },
    )


async def configure_telegram_webhook() -> None:
    if not WEBHOOK_URL:
        return
    if not WEBHOOK_SECRET:
        raise RuntimeError(
            "TELEGRAM_WEBHOOK_URL sozlangan, ammo "
            "TELEGRAM_WEBHOOK_SECRET yo‘q"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", WEBHOOK_SECRET):
        raise RuntimeError(
            "TELEGRAM_WEBHOOK_SECRET 32–256 ta harf, raqam, _ yoki - "
            "belgisidan iborat bo‘lishi kerak"
        )
    if not BOT_TOKEN or not ADMIN_ID or not ADMIN_ID.isdigit():
        raise RuntimeError(
            "Webhook uchun TELEGRAM_BOT_TOKEN va raqamli "
            "TELEGRAM_ADMIN_ID sozlanishi kerak"
        )
    await _telegram_request(
        "setWebhook",
        {
            "url": WEBHOOK_URL,
            "secret_token": WEBHOOK_SECRET,
            "allowed_updates": ["callback_query"],
        },
    )