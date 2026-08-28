from __future__ import annotations
import logging
from typing import TypeVar, Any
from pydantic import BaseModel, ValidationError
from guardrails.types import ValidationResult

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

MAX_INPUT_DISPLAY_CHARS = 40


def _format_input_for_diagnostic(val: Any) -> str:
    """Formatea el valor recibido truncando datos sensibles/largos para mitigar fuga de PII."""
    if val is None:
        return "None"
    val_str = repr(val)
    if len(val_str) > MAX_INPUT_DISPLAY_CHARS:
        return f"{val_str[:MAX_INPUT_DISPLAY_CHARS]}... (tipo: {type(val).__name__}, longitud: {len(val_str)})"
    return val_str


def validate_schema(data: dict | list | None, schema: type[T]) -> ValidationResult[T]:
    """
    Valida datos JSON contra un esquema Pydantic v2.
    Transforma errores de Pydantic en diagnóstico legible y estructurado sin eco masivo de PII.
    """
    if data is None:
        return ValidationResult(
            success=False,
            error_message="No se encontró JSON válido en la respuesta del modelo.",
            raw_json=None,
        )

    try:
        instance = schema.model_validate(data)
        return ValidationResult(success=True, data=instance, raw_json=data)
    except ValidationError as e:
        error_lines: list[str] = []
        for err in e.errors(include_url=False, include_input=True):
            loc = " → ".join(str(item) for item in err["loc"]) or "(raíz)"
            msg = err["msg"]
            input_val = _format_input_for_diagnostic(err.get("input", "N/A"))
            error_lines.append(f"- Campo '{loc}': {msg} (recibido: {input_val})")

        diagnostic = "\n".join(error_lines)
        logger.debug("Validación fallida:\n%s", diagnostic)
        return ValidationResult(
            success=False,
            error_message=diagnostic,
            raw_json=data,
        )
