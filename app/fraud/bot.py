import os

import httpx
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID")


async def send_telegram_message(text: str) -> bool:
    if not BOT_TOKEN or not ADMIN_ID:
        raise RuntimeError("Telegram konfiguratsiyasi topilmadi")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": ADMIN_ID,
        "text": text,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

        result = response.json()

        if not result.get("ok"):
            raise RuntimeError("Telegram xabar yuborilmadi")

        return True