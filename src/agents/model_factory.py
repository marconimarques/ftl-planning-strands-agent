"""Provider-agnostic model factory for Strands agents."""

from __future__ import annotations

import os
from typing import Any

from strands.models.anthropic import AnthropicModel
from strands.models.openai import OpenAIModel

# Registry: alias -> (provider, model_id)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # Provider-level aliases — used by /model anthropic|openai
    "anthropic":   ("anthropic", "claude-sonnet-4-6"),
    "openai":      ("openai",   "gpt-4o"),
    # Kept for internal use and data_expert fallback
    "haiku":       ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet":      ("anthropic", "claude-sonnet-4-6"),
    "opus":        ("anthropic", "claude-opus-4-8"),
    "gpt-4o-mini": ("openai",   "gpt-4o-mini"),
    "gpt-4o":      ("openai",   "gpt-4o"),
    "o3":          ("openai",   "o3"),
}

_ANTHROPIC_FAST_MODEL = "claude-haiku-4-5-20251001"
_OPENAI_FAST_MODEL = "gpt-4o-mini"

# Per-provider defaults per agent role.
# "reasoning" — OR Agent, Transportation Expert, Data Expert
# "agentic"   — Shock Response Agent (autonomous multi-step, benefits from o3)
_ANTHROPIC_REASONING_MODEL = "claude-sonnet-4-6"
_OPENAI_REASONING_MODEL    = "gpt-4o"
_OPENAI_AGENTIC_MODEL      = "gpt-4o"


def get_agent_model(provider: str, role: str = "reasoning") -> str:
    """Return the canonical model_id for this provider and agent role."""
    if provider == "openai":
        return _OPENAI_AGENTIC_MODEL if role == "agentic" else _OPENAI_REASONING_MODEL
    return _ANTHROPIC_REASONING_MODEL


def get_api_key(provider: str) -> str:
    if provider == "openai":
        return os.environ["OPENAI_API_KEY"]
    return os.environ["ANTHROPIC_API_KEY"]


def _openai_params(model_id: str, max_tokens: int) -> dict[str, Any]:
    # o1/o3/o4-series reasoning models use max_completion_tokens
    if model_id.startswith(("o1", "o3", "o4")):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def make_model(
    provider: str,
    model_id: str,
    api_key: str,
    max_tokens: int,
    system_prompt: str | None = None,
) -> tuple[Any, str | None]:
    """Return (strands_model, agent_system_prompt).

    For Anthropic with a system_prompt: bakes it into params with cache_control
    and returns None as agent_system_prompt.
    For OpenAI: returns a plain model and passes system_prompt through to Agent.
    """
    if provider == "openai":
        model = OpenAIModel(
            client_args={"api_key": api_key},
            model_id=model_id,
            params=_openai_params(model_id, max_tokens),
        )
        return model, system_prompt
    # Anthropic
    if system_prompt is not None:
        model = AnthropicModel(
            client_args={"api_key": api_key},
            model_id=model_id,
            max_tokens=max_tokens,
            params={"system": [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]},
        )
        return model, None
    model = AnthropicModel(
        client_args={"api_key": api_key},
        model_id=model_id,
        max_tokens=max_tokens,
    )
    return model, None
