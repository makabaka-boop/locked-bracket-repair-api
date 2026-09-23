"""Request/response schemas with strict validation.

Any extra field, wrong type, illegal character or bad token count makes
FastAPI answer 422 before the handler runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

Bracket = Literal["(", ")", "[", "]", "{", "}"]

MIN_TOKENS = 2
MAX_TOKENS = 160


class Token(BaseModel):
    model_config = ConfigDict(extra="forbid")

    char: Bracket
    locked: StrictBool


class RepairRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[Token] = Field(min_length=MIN_TOKENS, max_length=MAX_TOKENS)

    @field_validator("tokens")
    @classmethod
    def _even_length(cls, tokens: list[Token]) -> list[Token]:
        if len(tokens) % 2 != 0:
            raise ValueError("token count must be even")
        return tokens


class Change(BaseModel):
    index: int
    before: Bracket
    after: Bracket


class RepairResponse(BaseModel):
    status: Literal["OK", "NO_REPAIR"]
    repaired: str | None
    pairs: list[tuple[int, int]] | None
    changes: list[Change] | None
    cost: int | None
