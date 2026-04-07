from pydantic import BaseModel
from typing import Literal, Optional, Dict

class Observation(BaseModel):
    price: float
    rsi: float
    trend: str
    time_to_expiry: int
    position: str
    entry_price: Optional[float]
    balance: float
    equity: float

class Action(BaseModel):
    # Only allow valid actions defined by the Brain [cite: 193]
    action: Literal["BUY_CALL", "BUY_PUT", "HOLD", "EXIT"]

class Reward(BaseModel):
    value: float
    done: bool
    info: Dict