"""
llm_client.py — Anthropic API wrapper with disk cache.

Every call is keyed by SHA-256 of (model, system, user).  If a matching .json file
exists in .llm_cache/, the stored reply is returned without a network request.
This keeps API costs to a minimum and makes runs reproducible.

API key: read from ANTHROPIC_API_KEY environment variable; never written to any file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".llm_cache"


def _cache_dir() -> Path:
    cache = Path(os.environ.get("GNSS_LLM_CACHE", _DEFAULT_CACHE_DIR))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _cache_key(model: str, system: str, user: str) -> str:
    payload = json.dumps({"model": model, "system": system, "user": user}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def call(
    user_message: str,
    system_message: str = "",
    model: str = "claude-haiku-4-5",
    max_tokens: int = 1024,
) -> str:
    """
    Call Claude and return the text reply, using the disk cache.

    Raises EnvironmentError if ANTHROPIC_API_KEY is not set.
    The anthropic package is imported lazily so this module loads in --no-llm mode.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.  "
            "Export it in your shell: set ANTHROPIC_API_KEY=sk-ant-..."
        )

    key = _cache_key(model, system_message, user_message)
    cache_file = _cache_dir() / f"{key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["reply"]

    import anthropic  # lazy import — avoids hard dependency in --no-llm mode
    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_message}],
    }
    if system_message:
        kwargs["system"] = system_message

    response = client.messages.create(**kwargs)
    reply = response.content[0].text

    cache_file.write_text(
        json.dumps({"model": model, "system": system_message, "user": user_message, "reply": reply}, indent=2),
        encoding="utf-8",
    )
    return reply
