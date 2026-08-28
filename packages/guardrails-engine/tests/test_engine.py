import pytest
from pydantic import BaseModel, Field, field_validator
from guardrails.engine import SelfHealingEngine
from guardrails.types import TokenUsage
from guardrails.llm.types import Message, LLMResponse


class UserProfile(BaseModel):
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("El nombre no puede estar vacío")
        return v.strip()


class MockLLMClient:
    """Mock determinista que devuelve una secuencia de respuestas preconfiguradas."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = iter(responses)

    @property
    def supports_structured_output(self) -> bool:
        return False

    def complete(self, messages: list[Message]) -> LLMResponse:
        return next(self._responses)

    def complete_structured(self, messages: list[Message], json_schema: dict) -> LLMResponse:
        return self.complete(messages)


def _ok_response(text: str, tokens: int = 100) -> LLMResponse:
    return LLMResponse(
        text=text,
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=tokens, completion_tokens=tokens),
    )


def _truncated_response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        finish_reason="length",
        usage=TokenUsage(prompt_tokens=50, completion_tokens=50),
    )


class TestSelfHealingEngine:
    def test_success_first_attempt(self):
        client = MockLLMClient([_ok_response('{"name": "Juan", "age": 30}')])
        engine = SelfHealingEngine(llm_client=client, max_retries=2)
        result = engine.extract("Extrae datos de Juan, 30 años", UserProfile)
        assert result.success
        assert result.data is not None
        assert result.data.name == "Juan"
        assert result.attempts == 1
        assert result.total_tokens.total_tokens == 200

    def test_self_healing_second_attempt(self):
        """Falla en intento 1 (tipo incorrecto), éxito en intento 2."""
        client = MockLLMClient([
            _ok_response('{"name": "Juan", "age": "treinta"}'),  # Fallo: age es str
            _ok_response('{"name": "Juan", "age": 30}'),          # Corregido
        ])
        engine = SelfHealingEngine(llm_client=client, max_retries=2)
        result = engine.extract("Extrae datos", UserProfile)
        assert result.success
        assert result.data is not None
        assert result.data.age == 30
        assert result.attempts == 2
        assert result.total_tokens.total_tokens == 400

    def test_all_attempts_exhausted(self):
        """Falla persistente con errores diferentes -> agota reintentos controladamente."""
        client = MockLLMClient([
            _ok_response('{"name": "", "age": 30}'),       # Fallo: name vacío
            _ok_response('{"name": "J", "age": -5}'),      # Fallo: age negativo
            _ok_response('{"name": "J", "age": 200}'),     # Fallo: age > 150
        ])
        engine = SelfHealingEngine(llm_client=client, max_retries=2)
        result = engine.extract("Extrae datos", UserProfile)
        assert not result.success
        assert result.attempts == 3
        assert len(result.history) == 3

    def test_repeated_error_aborts_early(self):
        """Si el error es idéntico al anterior, aborta inmediatamente sin quemar tokens."""
        client = MockLLMClient([
            _ok_response('{"name": "Juan", "age": "viejo"}'),  # Fallo 1
            _ok_response('{"name": "Juan", "age": "viejo"}'),  # MISMO fallo
        ])
        engine = SelfHealingEngine(llm_client=client, max_retries=2)
        result = engine.extract("Extrae datos", UserProfile)
        assert not result.success
        assert result.attempts == 2  # No llega al intento 3
        assert result.error is not None
        assert "repetido" in result.error.lower()

    def test_truncated_response_no_retry(self):
        """Respuesta truncada por límite de tokens -> no reintentar."""
        client = MockLLMClient([_truncated_response('{"name": "Ju')])
        engine = SelfHealingEngine(llm_client=client, max_retries=2)
        result = engine.extract("Extrae datos", UserProfile)
        assert not result.success
        assert result.attempts == 1
        assert result.error is not None
        assert "truncada" in result.error.lower()

    def test_no_json_in_response_recovers(self):
        """Respuesta sin JSON recuperada en segundo intento."""
        client = MockLLMClient([
            _ok_response("Lo siento, no puedo extraer datos de ese texto."),
            _ok_response('{"name": "Juan", "age": 25}'),
        ])
        engine = SelfHealingEngine(llm_client=client, max_retries=2)
        result = engine.extract("Extrae datos", UserProfile)
        assert result.success
        assert result.data is not None
        assert result.data.name == "Juan"
        assert result.attempts == 2
