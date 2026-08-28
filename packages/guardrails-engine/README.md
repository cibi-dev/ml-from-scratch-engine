# 🛡️ Self-Healing Guardrails Engine

Ultra-lightweight, typed Python engine for deterministic LLM output extraction and self-healing validation using Pydantic v2.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2.7+-green.svg)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-purple.svg)](https://pytest.org/)

---

## 🚀 Key Features

- **Dual-Engine Architecture:** Primary execution via **Native Constrained Decoding** (Gemini `response_schema` / OpenAI `response_format: json_schema`) with automatic fallback to **Self-Healing Re-prompting** for non-supporting models.
- **4-Stage JSON Extractor:** Robust fallback cascade (Markdown fence -> Balanced brackets -> Direct loads -> Heuristic repair) handling real-world LLM anomalies.
- **Zero Heavy Dependencies:** No LangChain, no Instructor bloat, no complex graph frameworks. Pure, auditable Python (~400 lines).
- **Anti-Hallucination Guardrails:** Full Pydantic v2 schema enforcement with typed diagnostic error injection back into the LLM on validation failures.
- **Production Resilience:** HTTP timeout configuration (120s read), exponential backoff for 429/5xx, and early abortion on repeated failure loops.

---

## 📦 Quick Start

### 1. Installation

```bash
git clone https://github.com/cibi-dev/guardrails-engine.git
cd guardrails-engine
pip install -e .
```

### 2. Basic Usage

```python
from pydantic import BaseModel, Field
from guardrails import SelfHealingEngine, GeminiClient

class UserProfile(BaseModel):
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    skills: list[str]

# Initialize client and engine
client = GeminiClient(model="gemini-2.5-flash")
engine = SelfHealingEngine(llm_client=client, max_retries=2)

# Extract structured data
result = engine.extract(
    "Juan is a 25-year-old developer skilled in Python, Linux, and AI Agents.",
    UserProfile
)

if result.success:
    print(f"Name: {result.data.name}, Age: {result.data.age}")
    print(f"Attempts needed: {result.attempts}")
    print(f"Total tokens: {result.total_tokens.total_tokens}")
```

---

## 🧪 Testing

Run the full isolated unit test suite:

```bash
pytest tests/ -v
```

---

## 📄 License

MIT © 2026 Juan De Andrade
