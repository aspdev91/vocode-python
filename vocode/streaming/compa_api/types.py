from typing import List, TypedDict
from enum import Enum


class SenderType(Enum):
    BOT = "BOT"
    USER = "USER"


class MessageSender(TypedDict):
    type: SenderType


class StoredMessage(TypedDict):
    message: str
    messageSender: MessageSender
