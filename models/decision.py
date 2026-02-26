"""Pydantic model for LLM decision output validation and coercion.

Used after parse_llm_json to validate and coerce types (e.g. candidate_id to int).
On validation failure, the pipeline falls back to a safe action without retrying the LLM.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class LLMDecision(BaseModel):
    """Allowed shape for the LLM's single-step decision.

    action: one of done, click, type, select, navigate, scroll_down, scroll_up.
    candidate_id: required for click/type/select; coerced to int.
    text: required for type/select.
    url: required for navigate.
    """

    model_config = ConfigDict(extra="ignore")

    action: str
    candidate_id: Optional[int] = None
    text: Optional[str] = None
    url: Optional[str] = None

    @field_validator("candidate_id", mode="before")
    @classmethod
    def coerce_candidate_id(cls, v):
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None
