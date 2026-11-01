from pydantic import BaseModel, ConfigDict, Field
from decimal import Decimal


class TransactionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    sender_card_id: int = Field(gt=0)
    receiver_card_number: str = Field(pattern=r"^\d{16}$")
    amount: float = Field(ge=1000, allow_inf_nan=False)
    currency: str = "UZS"
    country: str = Field(min_length=2, max_length=100)
    city: str = Field(min_length=2, max_length=100)
    transactions_last_10_min: int = Field(default=0, ge=0)
    is_new_device: bool = False
    step: int = Field(default=1, ge=0)
    type: str = "TRANSFER"
    is_flagged_fraud: int = Field(
        default=0, ge=0, le=1, alias="isFlaggedFraud"
    )


class UserCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    phone: str = Field(min_length=9, max_length=13)
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8)
    balance: float = Field(default=0, ge=0)


class LoginRequest(BaseModel):
    phone: str = Field(max_length=13)
    password: str

class GetProfil(BaseModel):
    user_id: int
    first_name:str
    last_name:str
    phone:str

class GetUserCards(BaseModel):
    id: int
    last4: str
    display_pan: str
    card_number: str
    cardholder_name: str
    balance:Decimal
    currency: str
    is_default: bool


class GetUserBalance(BaseModel):
    total_balance: Decimal
    currency: str
    cards_count: int