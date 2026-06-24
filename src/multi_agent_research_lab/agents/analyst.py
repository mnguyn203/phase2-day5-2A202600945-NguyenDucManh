"""Analyst agent — turns research notes into structured insights."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a critical analyst. Your job is to analyse research notes and
extract structured insights that will help a writer create a high-quality response.

Output format (use exactly these sections):
## Key Claims
- Claim 1 [confidence: high/medium/low] — brief rationale
- ...

## Competing Viewpoints
- Viewpoint A vs Viewpoint B (if applicable)

## Evidence Gaps
- What is missing or weakly supported?

## Recommendation for Writer
- One sentence describing the main angle the writer should take."""

_USER_TEMPLATE = """Query: {query}
Target audience: {audience}

Research Notes:
{research_notes}

Analyse the notes above. Be critical and concise (150–250 words total).

IMPORTANT: Respond entirely in Vietnamese (Tiếng Việt), including all section headers and bullet points."""


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights for the Writer."""

    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Steps:
        1. Guard: research_notes must exist.
        2. Prompt LLM with query + notes → structured analysis.
        3. Record usage and trace.
        """
        logger.info("AnalystAgent starting | has_research=%s", state.research_notes is not None)

        # ── Guard ──────────────────────────────────────────────────────
        if not state.research_notes:
            state.record_error(self.name, "research_notes is empty — skipping analysis")
            state.analysis_notes = "[Analysis skipped: no research notes available]"
            return state

        # ── Build prompt ───────────────────────────────────────────────
        user_prompt = _USER_TEMPLATE.format(
            query=state.request.query,
            audience=state.request.audience,
            research_notes=state.research_notes,
        )

        # ── LLM call ───────────────────────────────────────────────────
        try:
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AgentExecutionError as exc:
            state.record_error(self.name, f"LLM call failed: {exc}")
            state.analysis_notes = f"[Analysis failed: {exc}]"
            return state

        state.analysis_notes = response.content
        state.record_llm_usage(response.input_tokens, response.output_tokens, response.cost_usd)

        # ── Record result and trace ────────────────────────────────────
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "analyst_done",
            {
                "analysis_length": len(response.content),
                "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                "cost_usd": response.cost_usd,
            },
        )

        logger.info(
            "AnalystAgent done | analysis=%d chars | tokens=%s | cost=$%s",
            len(response.content),
            response.input_tokens,
            f"{response.cost_usd:.6f}" if response.cost_usd else "N/A",
        )

        return state
