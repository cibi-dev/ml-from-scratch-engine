from __future__ import annotations
import json
import re
import logging

logger = logging.getLogger(__name__)

MAX_EXTRACTION_CHARS = 100_000


def extract_json_from_text(raw_text: str) -> dict | list | None:
    """
    Extrae un objeto o lista JSON de texto libre usando cascada de 4 estrategias:
    1. Fence markdown (```json ... ``` o ``` ... ```)
    2. Bloque balanceado más externo { } o [ ]
    3. json.loads directo sobre el texto completo (stripped)
    4. Reparación heurística (trailing commas, comillas simples)

    Retorna None si no se encuentra JSON parseable.
    """
    if not raw_text or not raw_text.strip():
        return None

    # Mitigación ReDoS: truncar entrada excesivamente grande
    if len(raw_text) > MAX_EXTRACTION_CHARS:
        logger.warning(
            "Texto de entrada excede %d caracteres (%d chars), truncando.",
            MAX_EXTRACTION_CHARS,
            len(raw_text),
        )
        raw_text = raw_text[:MAX_EXTRACTION_CHARS]

    # Estrategia 1: Fence markdown (```json ... ``` o ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw_text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        result = _try_parse(candidate)
        if result is not None:
            logger.debug("JSON extraído vía estrategia 1: Markdown fence")
            return result

    # Estrategia 2: Bloque balanceado más externo { } o [ ]
    result = _extract_balanced_block(raw_text)
    if result is not None:
        logger.debug("JSON extraído vía estrategia 2: Bloque balanceado")
        return result

    # Estrategia 3: json.loads directo sobre el texto completo (stripped)
    result = _try_parse(raw_text.strip())
    if result is not None:
        logger.debug("JSON extraído vía estrategia 3: json.loads directo")
        return result

    # Estrategia 4: Reparación heurística (trailing commas, comillas simples)
    repaired = _repair_json(raw_text)
    if repaired:
        result = _try_parse(repaired)
        if result is not None:
            logger.debug("JSON extraído vía estrategia 4: Reparación heurística")
            return result

    logger.warning("No se encontró JSON parseable en la respuesta")
    return None


def _try_parse(text: str) -> dict | list | None:
    """Intenta json.loads, retorna None en caso de error."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _extract_balanced_block(text: str) -> dict | list | None:
    """
    Busca el primer bloque balanceado { } o [ ] en una pasada O(n).
    Ignora llaves y corchetes dentro de strings JSON escapados.
    """
    first_brace = text.find('{')
    first_bracket = text.find('[')

    if first_brace == -1 and first_bracket == -1:
        return None

    # Determinar cuál delimitador aparece primero
    pairs: list[tuple[str, str]]
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        pairs = [('{', '}'), ('[', ']')]
    else:
        pairs = [('[', ']'), ('{', '}')]

    for open_char, close_char in pairs:
        start = text.find(open_char)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\' and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    result = _try_parse(candidate)
                    if result is not None:
                        return result
                    break

    return None


def _repair_json(text: str) -> str | None:
    """Reparaciones heurísticas conservadoras para trailing commas y comillas simples."""
    for open_c, close_c in [('{', '}'), ('[', ']')]:
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            # Trailing commas: ,} o ,]
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            # Comillas simples a dobles si no contiene dobles
            if "'" in repaired and '"' not in repaired:
                repaired = repaired.replace("'", '"')
            if repaired != candidate:
                return repaired
    return None
