from __future__ import annotations
import os
import logging
from typing import Any
import httpx
from guardrails.types import TokenUsage
from guardrails.llm.types import Message, LLMResponse
from guardrails.llm.http import LLM_TIMEOUT, request_with_retry

logger = logging.getLogger(__name__)


class GeminiClient:
    """Cliente para Gemini API (gemini-2.5-flash) vía REST con cabecera de autenticación segura."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no configurada")
        # Seguridad: la clave viaja en headers (x-goog-api-key), NUNCA en la URL
        # httpx respeta variables de entorno por defecto (trust_env=True); usar trust_env=False para aislar proxies/certificados de entorno.
        self._client = httpx.Client(
            timeout=LLM_TIMEOUT,
            headers={"x-goog-api-key": self.api_key},
        )

    @property
    def supports_structured_output(self) -> bool:
        return True

    def complete(self, messages: list[Message]) -> LLMResponse:
        return self._call(messages, json_schema=None)

    def complete_structured(self, messages: list[Message], json_schema: dict) -> LLMResponse:
        return self._call(messages, json_schema=json_schema)

    def _call(self, messages: list[Message], json_schema: dict | None) -> LLMResponse:
        url = f"{self.BASE_URL}/{self.model}:generateContent"

        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        generation_config: dict[str, Any] = {}
        if json_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = json_schema
        if generation_config:
            payload["generationConfig"] = generation_config

        response = request_with_retry(self._client, "POST", url, json=payload)
        return self._parse_gemini_response(response)

    def _parse_gemini_response(self, response: httpx.Response) -> LLMResponse:
        data = response.json()
        candidate = data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        finish_reason = candidate.get("finishReason", "STOP")

        usage_meta = data.get("usageMetadata", {})
        usage = TokenUsage(
            prompt_tokens=usage_meta.get("promptTokenCount", 0),
            completion_tokens=usage_meta.get("candidatesTokenCount", 0),
        )
        return LLMResponse(text=text, finish_reason=finish_reason, usage=usage)

    def close(self) -> None:
        """Cierra el cliente HTTP y libera los sockets."""
        self._client.close()

    def __enter__(self) -> GeminiClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class OpenAIClient:
    """Cliente para OpenAI API (gpt-4o-mini) vía REST."""

    BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY no configurada")
        # httpx respeta variables de entorno por defecto (trust_env=True); usar trust_env=False para aislar proxies/certificados de entorno.
        self._client = httpx.Client(
            timeout=LLM_TIMEOUT,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    @property
    def supports_structured_output(self) -> bool:
        return True

    def complete(self, messages: list[Message]) -> LLMResponse:
        return self._call(messages, json_schema=None)

    def complete_structured(self, messages: list[Message], json_schema: dict) -> LLMResponse:
        return self._call(messages, json_schema=json_schema)

    def _call(self, messages: list[Message], json_schema: dict | None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

        if json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_schema",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        response = request_with_retry(self._client, "POST", self.BASE_URL, json=payload)
        return self._parse_openai_response(response)

    def _parse_openai_response(self, response: httpx.Response) -> LLMResponse:
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason", "stop")

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )
        return LLMResponse(text=text, finish_reason=finish_reason, usage=usage)

    def close(self) -> None:
        """Cierra el cliente HTTP y libera los sockets."""
        self._client.close()

    def __enter__(self) -> OpenAIClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
