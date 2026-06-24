"""LangGraph workflow — orchestrates Supervisor + worker agents.

Graph topology:
    START → supervisor
    supervisor --(conditional)--> researcher | analyst | writer | END
    researcher → supervisor
    analyst    → supervisor
    writer     → supervisor
"""

from __future__ import annotations

import logging
import time

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node wrappers
# Langgraph nodes receive and return state dicts; we use Pydantic model_dump /
# model_validate to keep the rest of the code type-safe.
# ---------------------------------------------------------------------------

def _supervisor_node(state: ResearchState) -> ResearchState:
    return SupervisorAgent().run(state)


def _researcher_node(state: ResearchState) -> ResearchState:
    return ResearcherAgent().run(state)


def _analyst_node(state: ResearchState) -> ResearchState:
    return AnalystAgent().run(state)


def _writer_node(state: ResearchState) -> ResearchState:
    return WriterAgent().run(state)


def _route_after_supervisor(state: ResearchState) -> str:
    """Conditional edge: return the last route recorded by Supervisor."""
    if not state.route_history:
        return ROUTE_DONE
    return state.route_history[-1]


class MultiAgentWorkflow:
    """Builds and runs the LangGraph multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> object:
        """Create and compile a LangGraph StateGraph."""
        try:
            from langgraph.graph import END, StateGraph  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langgraph package not installed. Run: pip install 'langgraph>=0.2'"
            ) from exc

        graph: StateGraph = StateGraph(ResearchState)  # type: ignore[type-arg]

        # ── Nodes ──────────────────────────────────────────────────────
        graph.add_node("supervisor", _supervisor_node)
        graph.add_node("researcher", _researcher_node)
        graph.add_node("analyst",    _analyst_node)
        graph.add_node("writer",     _writer_node)

        # ── Entry point ────────────────────────────────────────────────
        graph.set_entry_point("supervisor")

        # ── Conditional edges from supervisor ──────────────────────────
        graph.add_conditional_edges(
            "supervisor",
            _route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst":    "analyst",
                "writer":     "writer",
                ROUTE_DONE:   END,
            },
        )

        # ── Workers always return to supervisor ────────────────────────
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst",    "supervisor")
        graph.add_edge("writer",     "supervisor")

        compiled = graph.compile()
        logger.debug("MultiAgentWorkflow | graph compiled successfully")
        return compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled graph and return the final ResearchState.

        Sets `state.start_time` before execution to enable latency tracking.
        """
        state.start_time = time.perf_counter()
        logger.info("MultiAgentWorkflow | starting | query='%s'", state.request.query)

        compiled = self.build()

        # LangGraph returns a dict-like snapshot; invoke with pydantic model
        result = compiled.invoke(state)

        # LangGraph may return a dict or a Pydantic model depending on version
        if isinstance(result, ResearchState):
            final_state = result
        elif isinstance(result, dict):
            final_state = ResearchState.model_validate(result)
        else:
            final_state = result  # type: ignore[assignment]

        elapsed = time.perf_counter() - (state.start_time or 0)
        logger.info(
            "MultiAgentWorkflow | finished | elapsed=%.2fs | routes=%s | errors=%d",
            elapsed,
            " → ".join(final_state.route_history),
            len(final_state.errors),
        )

        return final_state
