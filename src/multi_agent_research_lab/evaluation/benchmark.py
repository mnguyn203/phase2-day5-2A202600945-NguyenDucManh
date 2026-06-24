"""Benchmark skeleton for single-agent vs multi-agent."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

# Heuristic quality scoring weights
_MIN_ANSWER_WORDS = 100      # below this → quality penalty
_TARGET_ANSWER_WORDS = 400   # target word count for full score
_MAX_CITATIONS_SCORE = 3     # bonus points for citations (capped at 3)


def _score_quality(state: ResearchState) -> float:
    """Heuristic quality score 0–10.

    Scoring breakdown:
    - Up to 5 pts for answer length (relative to target word count)
    - Up to 2 pts for having research notes
    - Up to 3 pts for inline citations (capped)
    """
    score = 0.0

    answer = state.final_answer or ""
    word_count = len(answer.split())
    length_score = min(word_count / _TARGET_ANSWER_WORDS, 1.0) * 5.0
    score += length_score

    if state.research_notes:
        score += 1.0
    if state.analysis_notes:
        score += 1.0

    citation_score = min(state.citation_count, _MAX_CITATIONS_SCORE)
    score += citation_score

    return round(min(score, 10.0), 1)


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, and citation coverage.

    Args:
        run_name:  Human-readable label (e.g. "single-agent", "multi-agent").
        query:     Research query string.
        runner:    Callable that takes a query and returns a ResearchState.

    Returns:
        A tuple of (final_state, metrics).
    """
    logger.info("Benchmark starting | run='%s' | query='%s'", run_name, query)

    started = perf_counter()
    try:
        state = runner(query)
        success = True
    except Exception as exc:
        logger.error("Benchmark runner failed: %s", exc)
        # Build a minimal error state so the report still has a row
        from multi_agent_research_lab.core.schemas import ResearchQuery  # noqa: PLC0415
        from multi_agent_research_lab.core.state import ResearchState as _RS  # noqa: PLC0415
        state = _RS(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        success = False

    latency = perf_counter() - started

    quality = _score_quality(state) if success else 0.0
    notes_parts = []
    if state.errors:
        notes_parts.append(f"errors={len(state.errors)}")
    if state.citation_count:
        notes_parts.append(f"citations={state.citation_count}")
    notes_parts.append(f"words={len((state.final_answer or '').split())}")
    notes = "; ".join(notes_parts)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.total_cost_usd if state.total_cost_usd else None,
        quality_score=quality,
        notes=notes,
        response_content=state.final_answer,
    )

    logger.info(
        "Benchmark done | run='%s' | latency=%.2fs | cost=$%.6f | quality=%.1f/10",
        run_name,
        latency,
        state.total_cost_usd,
        quality,
    )

    return state, metrics
