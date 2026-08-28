from unittest.mock import MagicMock
import httpx
import pytest

from guardrails.llm.client import GeminiClient, OpenAIClient
from guardrails.llm.types import Message
from guardrails.llm.http import request_with_retry


def test_gemini_client_complete_and_structured(monkeypatch):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {"parts": [{"text": "{\"result\": \"ok\"}"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }

    monkeypatch.setattr("guardrails.llm.client.request_with_retry", lambda *args, **kwargs: mock_response)

    with GeminiClient(api_key="test-gemini-key") as client:
        assert client.supports_structured_output is True
        messages = [
            Message(role="system", content="Act as helper"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        resp = client.complete(messages)
        assert resp.text == "{\"result\": \"ok\"}"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5

        resp_struct = client.complete_structured(messages, json_schema={"type": "object"})
        assert resp_struct.text == "{\"result\": \"ok\"}"


def test_openai_client_complete_and_structured(monkeypatch):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "{\"status\": \"success\"}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8},
    }

    monkeypatch.setattr("guardrails.llm.client.request_with_retry", lambda *args, **kwargs: mock_response)

    with OpenAIClient(api_key="test-openai-key") as client:
        assert client.supports_structured_output is True
        messages = [
            Message(role="system", content="System instruction"),
            Message(role="user", content="Generate JSON"),
        ]
        resp = client.complete(messages)
        assert resp.text == "{\"status\": \"success\"}"
        assert resp.usage.prompt_tokens == 15

        resp_struct = client.complete_structured(messages, json_schema={"type": "object"})
        assert resp_struct.text == "{\"status\": \"success\"}"


def test_client_missing_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient(api_key=None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClient(api_key=None)


def test_http_retry_behavior():
    mock_client = MagicMock()
    resp_429 = MagicMock(spec=httpx.Response)
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "0"}
    resp_429.text = "Rate limited"

    resp_200 = MagicMock(spec=httpx.Response)
    resp_200.status_code = 200
    resp_200.json.return_value = {"status": "ok"}

    mock_client.request.side_effect = [resp_429, resp_200]

    resp = request_with_retry(mock_client, "POST", "https://api.test/retry", json={"test": 1}, max_retries=2)
    assert resp.status_code == 200
