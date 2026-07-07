from langgraph.graph import MessageState
from dataclasses import dataclass, field
from pydantic import BaseModel, field_validator
import os

from typing import TypedDict


class State(TypedDict):
    topic: str
    summary: str
    score: int


class State(BaseModel):
    topic: str
    score: int
    summary: str = ""

    @field_validator
    def score_positive(cls, v):
        if v < 0:
            raise ValueError("Score must be positive")


@dataclass
class State:
    topic: str = ""
    summary: str = ""
    message: list = field(default_factory=list)


class State(MessageState):
    user_name: str
    language: str
