import pytest
from guardrails.extractor import extract_json_from_text


class TestExtractor:
    def test_plain_json_object(self):
        assert extract_json_from_text('{"name": "Juan"}') == {"name": "Juan"}

    def test_plain_json_list(self):
        """Debe soportar listas en raíz."""
        assert extract_json_from_text('[{"id": 1}, {"id": 2}]') == [{"id": 1}, {"id": 2}]

    def test_markdown_fence(self):
        text = 'Aquí tienes el resultado:\n```json\n{"age": 25}\n```\nEso es todo.'
        assert extract_json_from_text(text) == {"age": 25}

    def test_embedded_in_conversation(self):
        text = 'La respuesta es: {"nombre": "Ana", "edad": 30}. Espero que te sirva.'
        result = extract_json_from_text(text)
        assert result == {"nombre": "Ana", "edad": 30}

    def test_trailing_comma(self):
        text = '{"name": "Juan", "age": 25,}'
        result = extract_json_from_text(text)
        assert result == {"name": "Juan", "age": 25}

    def test_no_json_returns_none(self):
        assert extract_json_from_text("No hay JSON aquí, lo siento.") is None

    def test_empty_string(self):
        assert extract_json_from_text("") is None

    def test_braces_inside_strings_not_confused(self):
        """Strings con {} no deben romper el extractor de llaves balanceadas."""
        text = '{"nota": "Usó {protección} en el incidente", "nivel": 3}'
        result = extract_json_from_text(text)
        assert isinstance(result, dict)
        assert result["nota"] == "Usó {protección} en el incidente"

    def test_single_quotes_repaired(self):
        text = "{'nombre': 'Juan', 'edad': 25}"
        result = extract_json_from_text(text)
        assert result == {"nombre": "Juan", "edad": 25}

    def test_nested_objects(self):
        text = '{"person": {"name": "Juan", "address": {"city": "Madrid"}}}'
        result = extract_json_from_text(text)
        assert isinstance(result, dict)
        assert result["person"]["address"]["city"] == "Madrid"

    def test_unicode_values(self):
        text = '{"nombre": "José María", "dirección": "Calle España ñ"}'
        result = extract_json_from_text(text)
        assert isinstance(result, dict)
        assert result["nombre"] == "José María"
