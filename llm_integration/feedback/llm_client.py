"""Provider-agnostic LLM completion (Anthropic or Gemini)."""

from __future__ import annotations

import logging
import os

import anthropic
import httpx
from dotenv import load_dotenv

_loaded = False
_claude: anthropic.Anthropic | None = None
_claude_model: str = ""

_log = logging.getLogger(__name__)
_PREVIEW = 2000

# Gemini REST (API key on aiplatform, same family as :streamGenerateContent curl)
_DEFAULT_GEMINI_BASE = "https://aiplatform.googleapis.com/v1/publishers/google/models"


def _env() -> None:
    global _loaded
    if not _loaded:
        load_dotenv()
        _loaded = True


def _provider() -> str:
    _env()
    p = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
    if p not in ("anthropic", "gemini"):
        raise ValueError("LLM_PROVIDER must be 'anthropic' or 'gemini'")
    return p


def _tok() -> int:
    return int(os.getenv("LLM_MAX_TOKENS", "40000"))


def _temp() -> float:
    return float(os.getenv("LLM_TEMPERATURE", "0.0"))


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    global _claude, _claude_model
    _env()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("ANTHROPIC_API_KEY not set")
    if _claude is None:
        _claude_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
        _claude = anthropic.Anthropic(api_key=api_key)
    parts: list[str] = []
    with _claude.messages.stream(
        model=_claude_model,
        max_tokens=_tok(),
        temperature=_temp(),
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            parts.append(text)
    return "".join(parts)


def _gemini_text_from_json(data: dict) -> str:
    chunks: list[str] = []
    for c in data.get("candidates") or []:
        content = c.get("content") or {}
        for p in content.get("parts") or []:
            t = p.get("text")
            if t:
                chunks.append(t)
    text = "".join(chunks).strip()
    if text:
        return text
    c0 = (data.get("candidates") or [None])[0]
    fr = (c0 or {}).get("finishReason")
    raise ValueError(f"Gemini empty text (finishReason={fr!r})")


def _call_gemini(system_prompt: str, user_message: str) -> str:
    _env()
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_api_key_here":
        raise ValueError("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    base = os.getenv("GEMINI_API_BASE", _DEFAULT_GEMINI_BASE).rstrip("/")
    url = f"{base}/{model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": _temp(),
            "maxOutputTokens": _tok(),
        },
    }
    timeout = float(os.getenv("LLM_TIMEOUT", "120"))
    with httpx.Client(timeout=timeout) as client:
        r = client.post(url, params={"key": key}, json=body)
    if r.status_code >= 400:
        raise ValueError(f"Gemini HTTP {r.status_code}: {r.text[:800]}")
    return _gemini_text_from_json(r.json())


def call_llm(system_prompt: str, user_message: str) -> str:
    prov = _provider()
    try:
        if prov == "gemini":
            result = _call_gemini(system_prompt, user_message)
        else:
            result = _call_anthropic(system_prompt, user_message)
    except Exception as e:
        _log.error("LLM request failed (%s): %s", prov, e)
        raise
    # tail = "…" if len(result) > _PREVIEW else ""
    _log.info(
        "LLM response [%s, %d chars]: %s%s",
        prov,
        len(result),
        "",
        "",  # result[:_PREVIEW], tail
    )
    return result
