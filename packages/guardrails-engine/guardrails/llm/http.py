from __future__ import annotations
import time
import random
import logging
from email.utils import parsedate_to_datetime
import httpx

logger = logging.getLogger(__name__)

LLM_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=120.0,
    write=10.0,
    pool=15.0,
)

HTTP_MAX_RETRIES = 3
HTTP_BASE_DELAY = 1.0
MAX_RETRY_DELAY: float = 60.0


def _parse_retry_after(header_val: str | None, default_delay: float) -> float:
    """Parsea Retry-After tolerando segundos enteros, decimales y fechas HTTP-Date. Retorno acotado a [0.0, MAX_RETRY_DELAY]."""
    if not header_val:
        return min(max(0.0, default_delay), MAX_RETRY_DELAY)
    header_str = header_val.strip()
    try:
        # Segundos directos (ej. "2" o "2.5")
        return min(max(0.0, float(header_str)), MAX_RETRY_DELAY)
    except ValueError:
        pass

    try:
        # Formato HTTP-Date (ej. "Wed, 21 Oct 2026 07:28:00 GMT")
        target_dt = parsedate_to_datetime(header_str)
        now_dt = target_dt.now(target_dt.tzinfo)
        diff = (target_dt - now_dt).total_seconds()
        return min(max(0.0, diff), MAX_RETRY_DELAY)
    except Exception:
        return min(max(0.0, default_delay), MAX_RETRY_DELAY)


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    max_retries: int = HTTP_MAX_RETRIES,
    base_delay: float = HTTP_BASE_DELAY,
    **kwargs,
) -> httpx.Response:
    """Ejecuta petición HTTP con reintentos, jitter y backoff exponencial para 429 y 5xx."""
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.request(method, url, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                calc_delay = min(base_delay * (2 ** attempt) + random.uniform(0.1, 0.5), MAX_RETRY_DELAY)
                delay = min(_parse_retry_after(response.headers.get("Retry-After"), calc_delay), MAX_RETRY_DELAY)

                logger.warning(
                    "HTTP %d en intento %d/%d, reintentando en %.2fs",
                    response.status_code,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                if attempt < max_retries:
                    time.sleep(delay)
                    continue
                response.raise_for_status()
            return response
        except httpx.TimeoutException as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0.1, 0.5), MAX_RETRY_DELAY)
                logger.warning(
                    "Timeout en intento %d, reintentando en %.2fs",
                    attempt + 1,
                    delay,
                )
                time.sleep(delay)
                continue
            raise

    raise last_exc or RuntimeError("Reintentos HTTP agotados")
