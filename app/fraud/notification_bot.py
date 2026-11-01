from app.fraud.bot import send_telegram_message


async def notify_transaction(transaction: dict) -> None:
    risk_score = float(transaction["risk_score"])

    if risk_score >= 0.8:
        risk_level = "HIGH"
        emoji = "🚨"
    elif risk_score >= 0.5:
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
        f"Risk bahosi: {risk_score:.2f}\n"
        f"Risk darajasi: {risk_level}"
    )
    if transaction.get("is_fraud"):
        message += "\nModel xulosasi: shubhali"

    await send_telegram_message(message)