"""Transaction ingestion and analyst-safe response schemas."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from backend.app.services.inference import prohibited_input_fields


AccountToken = Annotated[str, StringConstraints(min_length=4, max_length=128)]


class TransactionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=128)
    dataset: str = Field(default="online", min_length=1, max_length=64)
    event_time: datetime
    source_account_token: AccountToken
    destination_account_token: AccountToken
    amount: Decimal = Field(gt=0, max_digits=24, decimal_places=6)
    currency: str = Field(min_length=3, max_length=32)
    transaction_type: str = Field(min_length=1, max_length=64)
    raw_attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("event_time")
    @classmethod
    def event_time_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must include a timezone")
        return value

    @field_validator("raw_attributes")
    @classmethod
    def raw_attributes_must_be_label_free(
        cls, value: dict[str, str | int | float | bool | None]
    ) -> dict[str, str | int | float | bool | None]:
        forbidden = prohibited_input_fields(value)
        if forbidden:
            raise ValueError(f"evaluation-only fields are prohibited: {sorted(forbidden)}")
        return value


class TransactionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    event_id: str
    dataset: str
    event_time: datetime
    masked_source_account: str | None
    masked_destination_account: str | None
    amount: Decimal
    currency: str
    transaction_type: str
    created_at: datetime


def mask_account(value: str) -> str:
    value = str(value).strip()
    if len(value) <= 4:
        return "*" * len(value)
    return f"{'*' * min(8, len(value) - 4)}{value[-4:]}"


__all__ = ["TransactionCreate", "TransactionRead", "mask_account"]
