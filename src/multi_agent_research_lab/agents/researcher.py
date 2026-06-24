"""Researcher agent — collects sources and creates research notes."""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a meticulous research assistant. Your job is to synthesize
information from multiple web sources into concise, factual research notes.

Guidelines:
- Summarize the most important findings across all sources.
- Be objective; report what the sources say, not your own opinion.
- Mention which source (by index) supports each key point.
- Keep the total length between 200 and 350 words.
- Use plain prose, no markdown headers."""

_USER_TEMPLATE = """Query: {query}

Sources ({n_sources} documents):
{formatted_sources}

Write concise research notes covering the key findings relevant to the query.
Reference sources inline as [1], [2], etc.

IMPORTANT: Respond entirely in Vietnamese (Tiếng Việt)."""


def _format_sources(sources: list) -> str:
    parts: list[str] = []
    for i, src in enumerate(sources, start=1):
        title = src.title or "Untitled"
        url = f" ({src.url})" if src.url else ""
        snippet = (src.snippet or "")[:600]  # cap to avoid huge prompts
        parts.append(f"[{i}] {title}{url}\n{snippet}")
    return "\n\n".join(parts)


class ResearcherAgent(BaseAgent):
    """Collects sources via Tavily and creates concise research notes via LLM."""

    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._search = search or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Steps:
        1. Search web for the query using Tavily.
        2. Format sources into a prompt.
        3. Call LLM to synthesize research notes.
        4. Record token usage, agent result, and trace event.
        """
        logger.info("ResearcherAgent starting | query='%s'", state.request.query)

        # ── Step 1: Web search ─────────────────────────────────────────
        try:
            sources = self._search.search(
                query=state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources = sources
            logger.info("ResearcherAgent | fetched %d sources", len(sources))
        except AgentExecutionError as exc:
            state.record_error(self.name, f"Search failed: {exc}")
            logger.warning("ResearcherAgent | search failed, using empty sources: %s", exc)
            sources = []
            state.sources = []

        # ── Step 2: Build prompt ───────────────────────────────────────
        if sources:
            formatted = _format_sources(sources)
        else:
            formatted = "(No sources retrieved — answer from training knowledge only.)"

        user_prompt = _USER_TEMPLATE.format(
            query=state.request.query,
            n_sources=len(sources),
            formatted_sources=formatted,
        )

        # ── Step 3: LLM synthesis ──────────────────────────────────────
        try:
            response = self._llm.complete(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AgentExecutionError as exc:
            state.record_error(self.name, f"LLM call failed: {exc}")
            state.research_notes = f"[Research failed: {exc}]"
            return state

        state.research_notes = response.content
        state.record_llm_usage(response.input_tokens, response.output_tokens, response.cost_usd)

        # ── Step 4: Record result and trace ───────────────────────────
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "source_count": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher_done",
            {
                "source_count": len(sources),
                "notes_length": len(response.content),
                "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
                "cost_usd": response.cost_usd,
            },
        )

        logger.info(
            "ResearcherAgent done | notes=%d chars | tokens=%s | cost=$%s",
            len(response.content),
            response.input_tokens,
            f"{response.cost_usd:.6f}" if response.cost_usd else "N/A",
        )

        return state
