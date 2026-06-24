"""Tracing hooks — two-layer strategy.

Layer 1 — LangSmith auto-tracing:
    Set LANGSMITH_TRACING=true + LANGSMITH_API_KEY in .env.
    LangGraph automatically sends traces to LangSmith UI.

Layer 2 — JSON file tracer (always active, no external dependency):
    Writes a structured trace to reports/trace_<timestamp>.json
    so there's always a local audit trail.

This file intentionally avoids binding to one provider.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, strftime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Span context manager (reused internally + by agents)
# ---------------------------------------------------------------------------

@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context — records duration.

    Usage::

        with trace_span("researcher", {"query": q}) as span:
            # do work
            span["result"] = "..."
        # span["duration_seconds"] is now set
    """
    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
    }
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


# ---------------------------------------------------------------------------
# LangSmith setup helper
# ---------------------------------------------------------------------------

def configure_langsmith(api_key: str | None, project: str) -> bool:
    """Enable LangSmith auto-tracing if key is available.

    LangGraph picks up LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY
    environment variables automatically.

    Returns True if LangSmith was enabled.
    """
    if not api_key:
        logger.debug("LangSmith tracing skipped — LANGSMITH_API_KEY not set")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    logger.info("LangSmith tracing enabled | project='%s'", project)
    return True


# ---------------------------------------------------------------------------
# JSON file tracer
# ---------------------------------------------------------------------------

class JSONFileTracer:
    """Writes a complete trace to a JSON file in reports/.

    Each workflow run produces one file:
        reports/trace_YYYYMMDD_HHMMSS.json
    """

    def __init__(self, output_dir: str | Path = "reports") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace_events: list[dict[str, Any]], metadata: dict[str, Any]) -> Path:
        """Persist trace to disk and return the file path."""
        timestamp = strftime("%Y%m%d_%H%M%S")
        file_path = self._dir / f"trace_{timestamp}.json"

        payload = {
            "metadata": metadata,
            "events": trace_events,
        }

        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

        logger.info("JSONFileTracer | saved trace to %s", file_path)
        return file_path
