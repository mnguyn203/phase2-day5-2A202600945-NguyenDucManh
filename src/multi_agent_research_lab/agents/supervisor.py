"""Supervisor / router agent.

Decides which worker runs next using a rule-based policy.
Rule-based (not LLM) → deterministic, zero extra cost, easy to debug.
"""

from __future__ import annotations

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

# Sentinel value used as the route when the workflow is finished
ROUTE_DONE = "done"


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (sequential pipeline):
    ┌─────────────────────────────────────────────────────────┐
    │ research_notes is None  →  researcher                   │
    │ analysis_notes is None  →  analyst                      │
    │ final_answer   is None  →  writer                       │
    │ all populated            →  done                        │
    │                                                         │
    │ iteration >= max_iterations  →  force done (guardrail)  │
    └─────────────────────────────────────────────────────────┘
    """

    name = "supervisor"

    def __init__(self) -> None:
        self._settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Always returns the mutated state; never raises.
        """
        logger.info(
            "SupervisorAgent | iteration=%d/%d | research=%s | analysis=%s | final=%s",
            state.iteration,
            self._settings.max_iterations,
            "✓" if state.research_notes else "✗",
            "✓" if state.analysis_notes else "✗",
            "✓" if state.final_answer else "✗",
        )

        # ── Guardrail: max iterations ──────────────────────────────────
        if state.iteration >= self._settings.max_iterations:
            msg = (
                f"Max iterations ({self._settings.max_iterations}) reached. "
                "Forcing workflow to stop."
            )
            logger.warning("SupervisorAgent | %s", msg)
            state.record_error(self.name, msg)
            state.record_route(ROUTE_DONE)
            state.add_trace_event(
                "supervisor_routed",
                {"route": ROUTE_DONE, "reason": "max_iterations_exceeded"},
            )
            return state

        # ── Routing logic (sequential pipeline) ───────────────────────
        if state.research_notes is None:
            next_route = AgentName.RESEARCHER
            reason = "research_notes missing"
        elif state.analysis_notes is None:
            next_route = AgentName.ANALYST
            reason = "analysis_notes missing"
        elif state.final_answer is None:
            next_route = AgentName.WRITER
            reason = "final_answer missing"
        else:
            next_route = ROUTE_DONE  # type: ignore[assignment]
            reason = "all outputs populated"

        state.record_route(str(next_route))
        state.add_trace_event(
            "supervisor_routed",
            {
                "route": str(next_route),
                "reason": reason,
                "iteration": state.iteration,
            },
        )

        # ── Record as AgentResult ──────────────────────────────────────
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Routing to: {next_route}",
                metadata={"route": str(next_route), "reason": reason},
            )
        )

        logger.info("SupervisorAgent | routed to '%s' (%s)", next_route, reason)
        return state
