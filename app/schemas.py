from pydantic import BaseModel, ConfigDict, Field
from app.fraud.services import get_user_balance


class TransactionCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int
    amount: float
    currency: str
    country: str
    city: str
    transactions_last_10_min: int
    is_new_device: bool
    step: int
    type: str
    oldbalance_org: float = Field(alias="oldbalanceOrg")
    newbalance_orig: float = Field(alias="newbalanceOrig")
    oldbalance_dest: float = Field(alias="oldbalanceDest")
    newbalance_dest: float = Field(alias="newbalanceDest")
    is_flagged_fraud: int = Field(alias="isFlaggedFraud")

class UserCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    first_name: str
    last_name: str