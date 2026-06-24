"""Tests for agent implementations (replaces old TODO placeholder tests)."""

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.agents.supervisor import ROUTE_DONE
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _make_state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


class TestSupervisorRouting:
    def test_routes_to_researcher_when_no_notes(self) -> None:
        state = _make_state()
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "researcher"

    def test_routes_to_analyst_after_research(self) -> None:
        state = _make_state()
        state.research_notes = "Some research notes"
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "analyst"

    def test_routes_to_writer_after_analysis(self) -> None:
        state = _make_state()
        state.research_notes = "Some research notes"
        state.analysis_notes = "Some analysis"
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == "writer"

    def test_routes_to_done_when_all_populated(self) -> None:
        state = _make_state()
        state.research_notes = "Notes"
        state.analysis_notes = "Analysis"
        state.final_answer = "Answer"
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == ROUTE_DONE

    def test_max_iterations_guardrail(self) -> None:
        """When max_iterations reached, supervisor forces done and logs error."""
        state = _make_state()
        # Simulate many iterations
        state.iteration = 999
        result = SupervisorAgent().run(state)
        assert result.route_history[-1] == ROUTE_DONE
        assert len(result.errors) > 0

    def test_increments_iteration(self) -> None:
        state = _make_state()
        before = state.iteration
        SupervisorAgent().run(state)
        assert state.iteration == before + 1

    def test_adds_trace_event(self) -> None:
        state = _make_state()
        SupervisorAgent().run(state)
        names = [e["name"] for e in state.trace]
        assert "supervisor_routed" in names
