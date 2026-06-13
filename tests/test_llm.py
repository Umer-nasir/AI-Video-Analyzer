import pytest
from app import detect_provider

def test_detect_provider():
    assert detect_provider("sk-proj-...") == "OpenAI"
    assert detect_provider("gsk_...") == "Groq"
    assert detect_provider("AIza...") == "Gemini"
    assert detect_provider("sk-ant-api03-...") == "Anthropic"
    assert detect_provider("unknown_key") == "Unknown"
