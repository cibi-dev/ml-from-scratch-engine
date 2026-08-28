from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")
MAX_RAW_LOG_CHARS = 2000


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Conteo de tokens acumulable entre intentos."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class ValidationResult(Generic[T]):
    """Resultado de una validación de esquema contra datos JSON."""
    success: bool
    data: T | None = None
    error_message: str | None = None
    raw_json: dict | list | None = None


@dataclass
class AttemptRecord:
    """Registro de un intento individual de extracción."""
    attempt_number: int
    raw_response: str
    finish_reason: str
    token_usage: TokenUsage
    validation_error: str | None = None


@dataclass
class ExtractionResult(Generic[T]):
    """Resultado final de la extracción con auto-curación."""
    success: bool
    data: T | None = None
    attempts: int = 0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    history: list[AttemptRecord] = field(default_factory=list)
    error: str | None = None
