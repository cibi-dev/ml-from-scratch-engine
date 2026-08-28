import pytest
from pydantic import BaseModel, Field, field_validator, ConfigDict


class SampleUser(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Formato de email inválido")
        return v
