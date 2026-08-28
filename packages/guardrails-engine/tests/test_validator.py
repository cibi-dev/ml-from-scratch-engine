import pytest
from pydantic import BaseModel, Field, field_validator, ConfigDict
from guardrails.validator import validate_schema


class StrictUser(BaseModel):
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


class TestValidator:
    def test_valid_data(self):
        result = validate_schema({"name": "Juan", "age": 30, "email": "j@x.com"}, StrictUser)
        assert result.success
        assert result.data is not None
        assert result.data.name == "Juan"
        assert result.data.age == 30

    def test_wrong_type(self):
        result = validate_schema({"name": "Juan", "age": "treinta", "email": "j@x.com"}, StrictUser)
        assert not result.success
        assert result.error_message is not None
        assert "age" in result.error_message

    def test_missing_field(self):
        result = validate_schema({"name": "Juan", "age": 30}, StrictUser)
        assert not result.success
        assert result.error_message is not None
        assert "email" in result.error_message

    def test_extra_field_forbidden(self):
        result = validate_schema(
            {"name": "Juan", "age": 30, "email": "j@x.com", "extra": "nope"}, StrictUser
        )
        assert not result.success
        assert result.error_message is not None
        assert "extra" in result.error_message.lower()

    def test_field_validator_fails(self):
        result = validate_schema({"name": "Juan", "age": 30, "email": "invalid"}, StrictUser)
        assert not result.success
        assert result.error_message is not None
        assert "email" in result.error_message

    def test_none_data(self):
        result = validate_schema(None, StrictUser)
        assert not result.success
        assert result.error_message is not None
        assert "No se encontró JSON" in result.error_message

    def test_list_data_against_single_model(self):
        result = validate_schema([{"name": "Juan"}], StrictUser)
        assert not result.success

    def test_error_message_includes_received_value(self):
        result = validate_schema({"name": "Juan", "age": "viejo", "email": "j@x.com"}, StrictUser)
        assert not result.success
        assert result.error_message is not None
        assert "viejo" in result.error_message
