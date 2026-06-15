"""Lightweight intent classifier — routes queries to the correct pipeline."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel
from strands import Agent

from .model_factory import _ANTHROPIC_FAST_MODEL, _OPENAI_FAST_MODEL, get_api_key, make_model

_CLASSIFIER_PROMPT_PT = """\
Você classifica perguntas de um sistema de planejamento de frota de caminhões.

Retorne "shock_response" se o usuário está pedindo a melhor estratégia para
compensar uma deterioração operacional: redução de payload, aumento de custo,
queda de disponibilidade, redução de jornada, etc.
Frases típicas: "melhor reação", "como compensar", "o que fazer se",
"como conter", "como mitigar".

Retorne "what_if" para qualquer outra coisa: simulações, cenários hipotéticos,
cálculo de baseline, perguntas sobre a rede.

Retorne JSON com um único campo: query_type.
"""

_CLASSIFIER_PROMPT_EN = """\
You classify queries for a truck fleet planning system.

Return "shock_response" if the user is asking for the best strategy to offset
an operational deterioration: payload reduction, cost increase, availability
drop, reduced working hours, etc.
Typical phrasings: "best response to", "how to offset", "what to do if",
"how to contain", "how to mitigate".

Return "what_if" for everything else: simulations, what-if scenarios,
baseline calculations, network questions.

Return JSON with a single field: query_type.
"""


class _IntentResult(BaseModel):
    query_type: Literal["what_if", "shock_response"]


def create_classifier_agent(provider: str, api_key: str, language: str) -> Agent:
    """Create a reusable classifier agent. Uses the fast model for the given provider."""
    system_prompt = _CLASSIFIER_PROMPT_PT if language == "pt" else _CLASSIFIER_PROMPT_EN
    fast_model = _OPENAI_FAST_MODEL if provider == "openai" else _ANTHROPIC_FAST_MODEL
    model, agent_system_prompt = make_model(provider, fast_model, api_key, 512, system_prompt)
    return Agent(model=model, system_prompt=agent_system_prompt, tools=[], callback_handler=None)


def classify_intent(
    query: str, language: str, agent: Optional[Agent] = None
) -> Literal["what_if", "shock_response"]:
    """Classify a user query. Always uses Haiku regardless of the user's model setting.

    Pass a pre-built agent (from create_classifier_agent) to reuse the HTTP connection
    and benefit from system-prompt caching. Falls back to creating a fresh agent if
    no agent is provided (backward-compatible, no caching).
    """
    if agent is None:
        agent = create_classifier_agent("anthropic", get_api_key("anthropic"), language)
    else:
        agent.messages = []
    result = agent(query, structured_output_model=_IntentResult)
    structured = result.structured_output
    return structured.query_type if structured else "what_if"
