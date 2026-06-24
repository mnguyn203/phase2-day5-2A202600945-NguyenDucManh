"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# gpt-4o-mini pricing (USD per 1K tokens), as of mid-2025
_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "gpt-4o": {"input": 0.005000, "output": 0.015000},
    "gpt-3.5-turbo": {"input": 0.000500, "output": 0.001500},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a completion."""
    prices = _COST_PER_1K.get(model, _COST_PER_1K["gpt-4o-mini"])
    return (input_tokens / 1000) * prices["input"] + (output_tokens / 1000) * prices["output"]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient:
    """Provider-agnostic LLM client — backed by OpenAI.

    Features:
    - Automatic retry with exponential back-off (tenacity)
    - Token usage + cost tracking per call
    - Timeout respected from settings
    """

    def __init__(self, model: str | None = None) -> None:
        self._settings = get_settings()
        self._model = model or self._settings.openai_model
        self._client: Any = None  # lazy-init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai  # noqa: PLC0415
            except ImportError as exc:
                raise AgentExecutionError(
                    "openai package not installed. Run: pip install openai"
                ) from exc

            if not self._settings.openai_api_key:
                raise AgentExecutionError(
                    "OPENAI_API_KEY is not set. Check your .env file."
                )

            self._client = openai.OpenAI(
                api_key=self._settings.openai_api_key,
                timeout=float(self._settings.timeout_seconds),
            )
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Connects to OpenAI with retry logic and token cost tracking.
        """
        client = self._get_client()

        logger.debug("LLMClient.complete | model=%s | system_len=%d | user_len=%d",
                     self._model, len(system_prompt), len(user_prompt))

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            raise AgentExecutionError(f"LLM call failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else None
        output_tokens = response.usage.completion_tokens if response.usage else None
        cost = (
            _estimate_cost(self._model, input_tokens or 0, output_tokens or 0)
            if input_tokens is not None
            else None
        )

        logger.debug(
            "LLMClient.complete | input_tokens=%d | output_tokens=%d | cost=$%.6f",
            input_tokens or 0,
            output_tokens or 0,
            cost or 0.0,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model=self._model,
        )

    def complete_messages(self, messages: list[LLMMessage]) -> LLMResponse:
        """Complete from a list of LLMMessage objects (for multi-turn support)."""
        if not messages:
            raise ValueError("messages list cannot be empty")
        system = next((m.content for m in messages if m.role == "system"), "")
        user_parts = [m.content for m in messages if m.role != "system"]
        return self.complete(system_prompt=system, user_prompt="\n\n".join(user_parts))
