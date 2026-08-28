from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable
from guardrails.types import TokenUsage


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    text: str
    finish_reason: str  # "stop"/"STOP", "length"/"MAX_TOKENS", "safety"/"SAFETY"
    usage: TokenUsage

    @property
    def is_truncated(self) -> bool:
        """True si la respuesta fue cortada por agotar el límite de tokens."""
        return self.finish_reason.upper() in ("LENGTH", "MAX_TOKENS")


@runtime_checkable
class LLMClient(Protocol):
    """Protocolo que cualquier cliente LLM debe implementar."""

    def complete(self, messages: list[Message]) -> LLMResponse: ...

    @property
    def supports_structured_output(self) -> bool: ...

    def complete_structured(
        self,
        messages: list[Message],
        json_schema: dict,
    ) -> LLMResponse: ...
