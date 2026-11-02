from app.fraud.bot import (
    send_telegram_message,
    telegram_approval_enabled,
)


async def notify_transaction(transaction: dict) -> None:
    risk_score = float(transaction["risk_score"])
    model_risk_score = float(transaction["model_risk_score"])
    contextual_risk_score = float(transaction["contextual_risk_score"])
    fraud_threshold = float(transaction["fraud_threshold"])
    review_threshold = float(transaction["review_threshold"])
    model_inputs = transaction["model_inputs"]
    model_is_fraud = bool(transaction["is_fraud"])

    if model_is_fraud:
        risk_level = "HIGH"
        emoji = "🚨"
    elif risk_score >= review_threshold:
        risk_level = "MEDIUM"
        emoji = "⚠️"
    else:
        risk_level = "LOW"
        emoji = "🟢"

    message = (
        f"{emoji} TRANZAKSIYA\n\n"
        f"ID: {transaction['id']}\n"
        f"Holat: {transaction.get('transaction_status', transaction.get('status', 'Noma’lum'))}\n"
        f"Summa: {transaction['amount']} "
        f"{transaction['currency']}\n"
        f"Jo‘natuvchi: {transaction['sender_name']}\n"
        f"Telefon: {transaction['sender_phone']}\n"
        f"Jo‘natuvchi karta: {transaction['sender_card_number']}\n"
        f"Qabul qiluvchi: {transaction['recipient_name']}\n"
        f"Qabul qiluvchi karta: {transaction['receiver_card_number']}\n"
        f"Shahar/davlat: {transaction['city']}, {transaction['country']}\n"
        f"Qurilma: {transaction['device_name'] or 'Noma’lum'}\n"
        f"Yangi qurilma: {'Ha' if transaction['is_new_device'] else 'Yo‘q'}\n"
        f"Oxirgi 10 daqiqadagi oldingi tranzaksiyalar: "
        f"{transaction['transactions_last_10_min']}\n\n"
        "PREDICTION NATIJASI\n"
        f"Model ehtimoli: {model_risk_score:.1%}\n"
        f"Model fraud chegarasi: {fraud_threshold:.1%}\n"
        "Model xulosasi: "
        f"{'SHUBHALI' if model_is_fraud else 'shubhali deb topilmadi'}\n"
        f"Kontekst xavfi (qurilma/tezlik): {contextual_risk_score:.1%}\n"
        f"Yakuniy risk: {risk_score:.1%}\n"
        f"Tekshiruv chegarasi: {review_threshold:.1%}\n"
        f"Yakuniy qaror: {transaction['transaction_status']} ({risk_level})\n\n"
        "Modelga berilgan qiymatlar:\n"
        f"• Tranzaksiya turi: {model_inputs['type']}\n"
        f"• Soat qadami (step): {model_inputs['step']}\n"
        f"• Summa: {float(model_inputs['amount']):,.2f} "
        f"{transaction['currency']}\n"
        f"• Jo‘natuvchi oldingi balansi: "
        f"{float(model_inputs['oldbalanceOrg']):,.2f}\n"
        f"• Qabul qiluvchi oldingi balansi: "
        f"{float(model_inputs['oldbalanceDest']):,.2f}\n"
        f"• Manbadagi fraud belgisi: {model_inputs['isFlaggedFraud']}\n\n"
    )
    if telegram_approval_enabled():
        message += (
            "Avval foydalanuvchiga telefon qilib, shaxsini mustaqil "
            "tekshiring. Tasdiqlash tugmasi haqiqiy balanslardan pul o‘tkazadi."
        )
    else:
        message += (
            "Telegram webhook sozlanmagan: bir tugmali tasdiqlash faol emas. "
            "Telefon orqali tekshiruvni bajaring."
        )
    reasons = transaction.get("reasons", [])
    if reasons:
        message += "\n\nRisk signallari:"
        for reason in reasons[:5]:
            feature = reason.get("feature")
            description = reason.get("description")
            if feature:
                message += f"\n• {feature}: {description or reason.get('value')}"

    markup = None
    if telegram_approval_enabled():
        markup = {
            "inline_keyboard": [
                [
                    {
                        "text": "📞 Telefon raqamini ko‘rish",
                        "callback_data": f"contact:{transaction['id']}",
                    }
                ],
                [
                    {
                        "text": "✅ Shaxsni tekshirdim — o‘tkazish",
                        "callback_data": f"approve:{transaction['id']}",
                    }
                ],
            ]
        }

    await send_telegram_message(message, reply_markup=markup)