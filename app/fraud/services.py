from sqlalchemy import select
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User


async def get_user_balance(user_id:int, db:AsyncSession  = Depends(get_db)):
    balance = select(User.balance).where(user_id == User.id)
    result = await db.execute(balance)
    return result
