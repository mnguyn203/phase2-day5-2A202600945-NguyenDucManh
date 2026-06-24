"""Shared state for the multi-agent workflow.

Students should extend this file when adding new agents, outputs, or evaluation metrics.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from multi_agent_research_lab.core.schemas import AgentResult, ResearchQuery, SourceDocument


class ResearchState(BaseModel):
    """Single source of truth passed through the workflow.

    Design decisions:
    - All fields are optional/defaulted so state can be partially populated.
    - Token + cost fields are accumulated across all agent calls.
    - `trace` provides a human-readable audit log for debugging.
    - `errors` accumulates non-fatal issues; agents must NOT raise on soft failures.
    """

    # ── Core request ──────────────────────────────────────────────────
    request: ResearchQuery

    # ── Workflow control ──────────────────────────────────────────────
    iteration: int = 0
    route_history: list[str] = Field(default_factory=list)

    # ── Agent outputs ─────────────────────────────────────────────────
    sources: list[SourceDocument] = Field(default_factory=list)
    research_notes: str | None = None
    analysis_notes: str | None = None
    final_answer: str | None = None

    # ── Audit / observability ─────────────────────────────────────────
    agent_results: list[AgentResult] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # ── Benchmark / cost tracking ─────────────────────────────────────
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    start_time: float | None = None   # perf_counter() at workflow start

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def record_route(self, route: str) -> None:
        """Append next route and increment iteration counter."""
        self.route_history.append(route)
        self.iteration += 1

    def add_trace_event(self, name: str, payload: dict[str, Any]) -> None:
        """Append a structured trace event with a wall-clock timestamp."""
        self.trace.append(
            {
                "name": name,
                "timestamp": time.time(),
                "iteration": self.iteration,
                "payload": payload,
            }
        )

    def record_llm_usage(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        """Accumulate token and cost counters from a single LLM call."""
        self.total_input_tokens += input_tokens or 0
        self.total_output_tokens += output_tokens or 0
        self.total_cost_usd += cost_usd or 0.0

    def record_error(self, agent: str, message: str) -> None:
        """Log a non-fatal error with agent context."""
        entry = f"[{agent}] {message}"
        self.errors.append(entry)
        self.add_trace_event("error", {"agent": agent, "message": message})

    @property
    def total_tokens(self) -> int:
        """Total token usage (input + output) across all agents."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def citation_count(self) -> int:
        """Count inline citation markers like [1], [2], ... in final_answer."""
        if not self.final_answer:
            return 0
        import re  # noqa: PLC0415
        return len(re.findall(r"\[\d+\]", self.final_answer))

    @property
    def elapsed_seconds(self) -> float | None:
        """Return elapsed seconds since workflow start, or None if not started."""
        if self.start_time is None:
            return None
        import time as _time  # noqa: PLC0415
        return _time.perf_counter() - self.start_time
