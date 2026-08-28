from __future__ import annotations
import re
import logging
from typing import TypeVar
from pydantic import BaseModel

from guardrails.types import (
    ExtractionResult,
    AttemptRecord,
    TokenUsage,
    MAX_RAW_LOG_CHARS,
)
from guardrails.extractor import extract_json_from_text
from guardrails.validator import validate_schema
from guardrails.prompts import (
    build_system_prompt,
    build_correction_prompt,
    wrap_untrusted_input,
)
from guardrails.llm.types import LLMClient, Message

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

MAX_USER_INPUT_CHARS = 100_000


_API_KEY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"AIza[0-9A-Za-z-_]{35}"), "[REDACTED_GEMINI_KEY]"),
    (re.compile(r"sk-ant-[0-9A-Za-z_\-]{20,}"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"sk-[A-Za-z0-9-_]{20,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"(?i)bearer\s+[0-9A-Za-z_\-\.=/+]{20,}"), "Bearer [REDACTED]"),
    (re.compile(r"key=[^&\s]+"), "key=[REDACTED]"),
]


def _sanitize_error_string(err_str: str) -> str:
    """Sanitiza mensajes de error y excepciones para evitar fuga accidental de API keys o URLs con secretos."""
    sanitized: str = err_str
    for pattern, replacement in _API_KEY_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


class SelfHealingEngine:
    """Motor de extracción estructurada con auto-curación y soporte dual native/fallback."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_retries: int = 2,
        prefer_native: bool = True,
    ):
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.max_attempts = max_retries + 1
        self.prefer_native = prefer_native

    def extract(self, prompt: str, schema: type[T]) -> ExtractionResult[T]:
        """
        Extrae datos estructurados de un LLM con auto-curación en bucle determinista.

        Args:
            prompt: El prompt del usuario con el contexto o texto libre a extraer.
            schema: Clase Pydantic v2 (BaseModel) que define el esquema.

        Returns:
            ExtractionResult con la instancia tipada o el error final.
        """
        if not prompt or not prompt.strip():
            return ExtractionResult(
                success=False,
                attempts=0,
                error="El prompt de entrada está vacío.",
            )

        if len(prompt) > MAX_USER_INPUT_CHARS:
            logger.warning("Prompt de usuario excede %d chars, truncando.", MAX_USER_INPUT_CHARS)
            prompt = prompt[:MAX_USER_INPUT_CHARS]

        schema_json = schema.model_json_schema()
        total_tokens = TokenUsage()
        history: list[AttemptRecord] = []
        previous_error: str | None = None
        use_native = self.prefer_native and self.llm_client.supports_structured_output

        # Contención segura de prompt injection en intento inicial
        wrapped_user_prompt = wrap_untrusted_input(prompt)

        for attempt in range(1, self.max_attempts + 1):
            if attempt == 1:
                messages = [
                    Message(role="system", content=build_system_prompt(schema_json)),
                    Message(role="user", content=wrapped_user_prompt),
                ]
            else:
                # Re-prompting limpio con corrección
                messages = [
                    Message(role="system", content=build_system_prompt(schema_json)),
                    Message(
                        role="user",
                        content=build_correction_prompt(previous_error or "", schema_json),
                    ),
                ]

            try:
                if use_native and attempt == 1:
                    response = self.llm_client.complete_structured(messages, schema_json)
                else:
                    response = self.llm_client.complete(messages)
            except Exception as e:
                clean_err = _sanitize_error_string(str(e))
                logger.error("Error de red/API en intento %d: %s", attempt, clean_err)
                history.append(
                    AttemptRecord(
                        attempt_number=attempt,
                        raw_response=clean_err[:MAX_RAW_LOG_CHARS],
                        finish_reason="ERROR",
                        token_usage=TokenUsage(),
                        validation_error=f"Error de API: {clean_err}",
                    )
                )
                continue

            total_tokens = total_tokens + response.usage

            # Verificar truncamiento
            if response.is_truncated:
                logger.warning(
                    "Respuesta truncada (finish_reason=%s) en intento %d",
                    response.finish_reason,
                    attempt,
                )
                record = AttemptRecord(
                    attempt_number=attempt,
                    raw_response=response.text[:MAX_RAW_LOG_CHARS],
                    finish_reason=response.finish_reason,
                    token_usage=response.usage,
                    validation_error="Respuesta truncada por max_output_tokens",
                )
                history.append(record)
                return ExtractionResult(
                    success=False,
                    attempts=attempt,
                    total_tokens=total_tokens,
                    history=history,
                    error="Respuesta truncada por max_output_tokens. Incrementar límite de tokens o simplificar esquema.",
                )

            # Extraer JSON de la respuesta
            json_data = extract_json_from_text(response.text)

            # Validar contra esquema Pydantic v2
            validation = validate_schema(json_data, schema)

            record = AttemptRecord(
                attempt_number=attempt,
                raw_response=response.text[:MAX_RAW_LOG_CHARS],
                finish_reason=response.finish_reason,
                token_usage=response.usage,
                validation_error=validation.error_message,
            )
            history.append(record)

            if validation.success:
                logger.info(
                    "Extracción exitosa en intento %d/%d (tokens: %d)",
                    attempt,
                    self.max_attempts,
                    total_tokens.total_tokens,
                )
                return ExtractionResult(
                    success=True,
                    data=validation.data,
                    attempts=attempt,
                    total_tokens=total_tokens,
                    history=history,
                )

            # Detección de bucle de error idéntico
            current_error = validation.error_message
            if current_error == previous_error:
                logger.warning(
                    "Error idéntico repetido en intento %d, abortando.",
                    attempt,
                )
                return ExtractionResult(
                    success=False,
                    attempts=attempt,
                    total_tokens=total_tokens,
                    history=history,
                    error=f"Error repetido sin progreso: {current_error}",
                )

            previous_error = current_error
            logger.info(
                "Intento %d/%d fallido, activando self-healing. Error: %s",
                attempt,
                self.max_attempts,
                current_error,
            )

        logger.error(
            "Todos los %d intentos fallaron. Último error: %s",
            self.max_attempts,
            previous_error,
        )
        return ExtractionResult(
            success=False,
            attempts=self.max_attempts,
            total_tokens=total_tokens,
            history=history,
            error=previous_error,
        )

    def close(self) -> None:
        """Cierra el cliente subyacente si posee método close."""
        if hasattr(self.llm_client, "close") and callable(self.llm_client.close):
            self.llm_client.close()

    def __enter__(self) -> SelfHealingEngine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
