"""Writer agent — produces the final answer from research and analysis notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert technical writer. Your job is to synthesise research
notes and analysis into a clear, engaging, well-structured response.

Writing guidelines:
- Target the specified audience level.
- Write approximately 450–550 words.
- Start with a concise executive summary (2–3 sentences).
- Use markdown formatting: headers (##), bullet points where helpful.
- Cite sources inline as [1], [2], etc., wherever claims are made.
- End with a short "Key Takeaways" section (3–5 bullets).
- Do NOT invent facts; only use information from the notes provided."""

_USER_TEMPLATE = """Query: {query}
Target audience: {audience}

Research Notes:
{research_notes}

Analysis:
{analysis_notes}

Sources available (for citation):
{sources_list}

Write the final response following the guidelines above.

IMPORTANT: Respond entirely in Vietnamese (Tiếng Việt), including all headers, bullets, and the Key Takeaways section."""


def _format_sources_list(state: ResearchState) -> str:
    if not state.sources:
        return "(No external sources retrieved)"
    lines = [f"[{i}] {s.title} — {s.url or 'no URL'}" for i, s in enumerate(state.sources, 1)]
    return "\n".join(lines)


class WriterAgent(BaseAgent):
    """Produces final answer by synthesising all previous agent outputs."""

    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Steps:
        1. Guard: need at least research_notes.
        2. Build rich prompt combining research + analysis + sources.
        3. Call LLM.
        4. Record usage and trace.
        """
        logger.info(
            "WriterAgent starting | has_research=%s | has_analysis=%s",
            state.research_notes is not None,
            state.analysis_notes is not None,
        )

        # ── Guard ──────────────────────────────────────────────────────
        if not state.research_notes:
            state.record_error(self.name, "research_notes missing — cannot write final answer")
            state.final_answer = "[Writing failed: no research notes available]"
            return state

        # Use placeholder if analysis was skipped
        analysis = state.analysis_notes or "(Analysis not available — write based on research only)"

        # ── Build prompt ───────────────────────────────────────────────
        user_prompt = _USER_TEMPLATE.format(
            query=state.request.query,
            audience=state.request.audience,
            research_notes=state.research_notes,
            analysis_notes=analysis,
            sources_list=_format_sources_list(state),
        )

        # ── LLM call ───────────────────────────────────────────────────
        try:
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AgentExecutionError as exc:
            state.record_error(self.name, f"LLM call failed: {exc}")
            state.final_answer = f"[Writing failed: {exc}]"
            return state

        state.final_answer = response.content
        state.record_llm_usage(response.input_tokens, response.output_tokens, response.cost_usd)

        # ── Record result and trace ────────────────────────────────────
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "citation_count": state.citation_count,
                },
            )
        )
        state.add_trace_event(
            "writer_done",
            {
                "answer_length": len(response.content),
                "citation_count": state.citation_count,
                "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                "cost_usd": response.cost_usd,
            },
        )

        logger.info(
            "WriterAgent done | answer=%d chars | citations=%d | cost=$%s",
            len(response.content),
            state.citation_count,
            f"{response.cost_usd:.6f}" if response.cost_usd else "N/A",
        )

        return state
