"""Robust JSON parsing for LLM response text.

Handles markdown code fences, preamble text, and trailing commentary
that LLMs commonly add around JSON output. Optionally validates and
coerces the parsed dict via a Pydantic model.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from models.decision import LLMDecision


def parse_llm_json(content: str) -> dict:
    """Parse a JSON dict from LLM response text.

    Three-phase approach:
      1. Fast path -- try ``json.loads`` directly.
      2. Fence stripping -- remove markdown ```json / ``` wrappers.
      3. Object extraction -- find first ``{`` and last ``}``.

    Returns:
        A Python dict parsed from the JSON content.

    Raises:
        ValueError: If no valid JSON dict can be extracted.
    """
    raw = content.strip()
    if not raw:
        raise ValueError("LLM returned non-JSON: (empty string)")

    # --- Fast path: pure JSON ---
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # --- Fence stripping ---
    s = raw
    if s.startswith("```"):
        if s.startswith("```json"):
            s = s[7:]
        else:
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Object extraction: first { to last } ---
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(f"LLM returned non-JSON: {raw[:200]}")


def normalize_decision(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and coerce the parsed LLM decision dict.

    Ensures required 'action' is present and coerces candidate_id to int.
    Ignores extra keys. Returns a dict suitable for build_action, or None
    on validation failure (caller should use fallback without retrying LLM).
    """
    if not isinstance(raw, dict) or "action" not in raw:
        return None
    try:
        decision = LLMDecision.model_validate(raw)
        return decision.model_dump(exclude_none=False)
    except ValidationError:
        return None
