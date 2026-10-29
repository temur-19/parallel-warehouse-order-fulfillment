from pydantic import BaseModel, ConfigDict, Field
from app.fraud.services import get_user_balance


class TransactionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    sender_card_number: str = Field(pattern=r"^\d{16}$")
    receiver_card_number: str = Field(pattern=r"^\d{16}$")
    amount: float = Field(ge=1000, allow_inf_nan=False)
    currency: str
    country: str
    city: str
    transactions_last_10_min: int
    is_new_device: bool
    step: int
    type: str
    is_flagged_fraud: int = Field(alias="isFlaggedFraud")

class UserCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    phone:str = Field(max_length=13)
    first_name: str
    last_name: str
    password: str = Field(min_length=8)
    balance: float = Field(default=0, ge=0)


class LoginRequest(BaseModel):
    phone:str = Field(max_length=13)
    password: str