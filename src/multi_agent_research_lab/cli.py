"""Command-line entrypoint for the lab starter."""

from __future__ import annotations

import os
import sys
import time
from typing import Annotated

# Đảm bảo stdout/stderr hỗ trợ UTF-8 trên Windows (tiếng Việt)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
os.environ.setdefault("PYTHONUTF8", "1")

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import save_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import JSONFileTracer, configure_langsmith

app = typer.Typer(help="Multi-Agent Research Lab — Lab 20")
console = Console()


# ---------------------------------------------------------------------------
# Init helpers
# ---------------------------------------------------------------------------

def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_langsmith(settings.langsmith_api_key, settings.langsmith_project)


def _run_single_agent(query: str) -> ResearchState:
    """Single-agent baseline: one LLM call that does everything."""
    from multi_agent_research_lab.core.schemas import ResearchQuery as _RQ  # noqa: PLC0415
    from multi_agent_research_lab.services.llm_client import LLMClient  # noqa: PLC0415

    system_prompt = (
        "You are an expert research assistant. Given a query, write a thorough ~500-word "
        "response covering key facts, analysis, and a conclusion. "
        "Use markdown formatting with headers and bullet points. "
        "Respond entirely in Vietnamese (Tiếng Việt)."
    )
    user_prompt = f"Research query: {query}"

    state = ResearchState(request=_RQ(query=query))
    state.start_time = time.perf_counter()

    llm = LLMClient()
    response = llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)

    state.final_answer = response.content
    state.record_llm_usage(response.input_tokens, response.output_tokens, response.cost_usd)
    state.add_trace_event("single_agent_done", {
        "tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
        "cost_usd": response.cost_usd,
    })

    return state


def _run_multi_agent(query: str) -> ResearchState:
    """Multi-agent pipeline: Supervisor → Researcher → Analyst → Writer."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


def _print_metrics_table(all_metrics: list[BenchmarkMetrics]) -> None:
    table = Table(title="Benchmark Comparison", show_header=True, header_style="bold magenta")
    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Notes")

    for m in all_metrics:
        cost = "N/A" if m.estimated_cost_usd is None else f"${m.estimated_cost_usd:.6f}"
        quality = "N/A" if m.quality_score is None else f"{m.quality_score:.1f}/10"
        table.add_row(m.run_name, f"{m.latency_seconds:.2f}", cost, quality, m.notes)

    console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (one LLM call)."""
    _init()
    console.print(Panel.fit("[bold]Single-Agent Baseline[/bold]", style="cyan"))

    started = time.perf_counter()
    try:
        state = _run_single_agent(query)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    elapsed = time.perf_counter() - started

    if state.final_answer:
        console.print(Markdown(state.final_answer))

    console.print(
        Panel.fit(
            f"Tokens: {state.total_tokens}  |  "
            f"Cost: ${state.total_cost_usd:.6f}  |  "
            f"Elapsed: {elapsed:.2f}s",
            title="Stats",
            style="green",
        )
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save_trace: Annotated[bool, typer.Option("--save-trace/--no-save-trace")] = True,
) -> None:
    """Run the full multi-agent workflow (Supervisor + Researcher + Analyst + Writer)."""
    _init()
    console.print(Panel.fit("[bold]Multi-Agent Workflow[/bold]", style="magenta"))

    started = time.perf_counter()
    try:
        state = _run_multi_agent(query)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    elapsed = time.perf_counter() - started

    # Print route history
    if state.route_history:
        route_str = " → ".join(state.route_history)
        console.print(f"\n[dim]Route: {route_str}[/dim]")

    # Print errors (if any)
    if state.errors:
        for err in state.errors:
            console.print(f"[yellow]⚠ {err}[/yellow]")

    # Print final answer
    if state.final_answer:
        console.print("\n")
        console.print(Markdown(state.final_answer))

    console.print(
        Panel.fit(
            f"Sources: {len(state.sources)}  |  "
            f"Tokens: {state.total_tokens}  |  "
            f"Cost: ${state.total_cost_usd:.6f}  |  "
            f"Citations: {state.citation_count}  |  "
            f"Elapsed: {elapsed:.2f}s",
            title="Stats",
            style="green",
        )
    )

    # Save JSON trace
    if save_trace and state.trace:
        tracer = JSONFileTracer()
        trace_path = tracer.save(
            trace_events=state.trace,
            metadata={
                "query": query,
                "route_history": state.route_history,
                "total_tokens": state.total_tokens,
                "total_cost_usd": state.total_cost_usd,
                "elapsed_seconds": elapsed,
                "errors": state.errors,
            },
        )
        console.print(f"\n[dim]Trace saved → {trace_path}[/dim]")


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option(
            "--query", "-q",
            help="Research query to benchmark",
        ),
    ] = "What is GraphRAG and how does it improve over standard RAG?",
) -> None:
    """Run both modes, compare metrics, and save benchmark_report.md."""
    _init()
    console.print(Panel.fit("[bold]Benchmark: Single-Agent vs Multi-Agent[/bold]", style="yellow"))
    console.print(f"[dim]Query: {query}[/dim]\n")

    all_metrics: list[BenchmarkMetrics] = []
    all_states: list[ResearchState] = []

    # ── Single-agent ────────────────────────────────────────────────
    console.print("[cyan]Running single-agent baseline...[/cyan]")
    try:
        state_s, metrics_s = run_benchmark("single-agent", query, _run_single_agent)
        all_metrics.append(metrics_s)
        all_states.append(state_s)
        console.print(f"  ✓ Done in {metrics_s.latency_seconds:.2f}s")
    except Exception as exc:
        console.print(f"  [red]✗ Failed: {exc}[/red]")

    # ── Multi-agent ─────────────────────────────────────────────────
    console.print("[magenta]Running multi-agent workflow...[/magenta]")
    try:
        state_m, metrics_m = run_benchmark("multi-agent", query, _run_multi_agent)
        all_metrics.append(metrics_m)
        all_states.append(state_m)
        console.print(f"  ✓ Done in {metrics_m.latency_seconds:.2f}s")

        # Save trace for multi-agent run
        if state_m.trace:
            tracer = JSONFileTracer()
            tracer.save(
                trace_events=state_m.trace,
                metadata={
                    "run": "benchmark-multi-agent",
                    "query": query,
                    "route_history": state_m.route_history,
                },
            )
    except Exception as exc:
        console.print(f"  [red]✗ Failed: {exc}[/red]")

    # ── Print table ──────────────────────────────────────────────────
    console.print()
    _print_metrics_table(all_metrics)

    # ── Save report ──────────────────────────────────────────────────
    if all_metrics:
        report_path = save_report(all_metrics, query=query)
        console.print(f"\n[green]✓ Report saved → {report_path}[/green]")


if __name__ == "__main__":
    app()
