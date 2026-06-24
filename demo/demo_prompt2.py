"""
Demo script: Prompt 2 — Research Briefing on Multi-Agent LLMs
==============================================================
Showcase demo cho thầy giáo — 5 điểm bonus.

Prompt 2 gốc:
    "Do multi-agent LLM systems actually outperform single-agent systems
    on complex tasks?"

Chạy:  python demo/demo_prompt2.py
Output: reports/demo_prompt2_result.md  +  reports/trace_*.json

Chiến lược:
    - Monkey-patch _SYSTEM_PROMPT của Analyst và Writer TRƯỚC khi
      khởi tạo bất kỳ class nào → code chính hoàn toàn không bị đụng.
    - Dùng một query tối ưu cho cả Tavily search và LLM context.
    - Analyst override → xử lý đúng 8 tasks + 4 constraints của Prompt 2.
    - Writer override  → output đúng 7-section format, như speaking notes.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── UTF-8 cho tiếng Việt trên Windows ────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONUTF8", "1")

# ── Load .env ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# =============================================================================
# BƯỚC 1 — Monkey-patch prompts TRƯỚC KHI import các class agent
# Kỹ thuật: module-level variables (_SYSTEM_PROMPT, _USER_TEMPLATE) là mutable.
# Gán giá trị mới trước khi WorkerAgent.__init__ chạy → agents dùng prompt mới.
# =============================================================================

import multi_agent_research_lab.agents.researcher as _researcher_mod  # noqa: E402
import multi_agent_research_lab.agents.analyst as _analyst_mod         # noqa: E402
import multi_agent_research_lab.agents.writer as _writer_mod           # noqa: E402

# ─── Researcher: tập trung vào bằng chứng for/against và confounders ─────────
_researcher_mod._SYSTEM_PROMPT = """\
You are a meticulous research assistant helping a PhD student prepare a research briefing.
Your job is to synthesize information from multiple web sources into factual research notes.

IMPORTANT FOCUS: The topic is whether multi-agent LLM systems truly outperform single-agent
systems. Pay special attention to:
- Empirical benchmark results (include exact numbers / percentages when available)
- Which paper/source supports multi-agent superiority vs. which raises doubts
- Any mention of token budget, prompt engineering, or self-reflection as confounding factors
- Methodology quality: was the comparison controlled and fair?
- Named systems mentioned: AutoGen, MetaGPT, AgentBench, CrewAI, BabyAGI, etc.

Guidelines:
- Report ONLY what the sources actually say — not your own opinion.
- Cite source index inline as [1], [2], etc. for every key finding.
- Keep total length 300–450 words.
- Use plain prose — no markdown headers.
- If a source is inconclusive or methodologically weak, flag it explicitly."""

_researcher_mod._USER_TEMPLATE = """\
Query: {query}

Sources ({n_sources} documents):
{formatted_sources}

Write structured research notes covering:
1. Key empirical findings (with numbers if available)
2. Which sources support multi-agent, which challenge it
3. Any confounding factors mentioned (tokens, prompts, self-reflection)
4. Methodology quality observations

Reference sources inline as [1], [2], etc.
IMPORTANT: Respond entirely in Vietnamese (Tiếng Việt)."""

# ─── Analyst: xử lý Tasks 2-7 và Constraints A-D của Prompt 2 ─────────────
_analyst_mod._SYSTEM_PROMPT = """\
You are a critical analyst helping a PhD student prepare speaking notes for a research group meeting.
Based on the research notes provided, produce STRUCTURED ANALYSIS — not a descriptive summary.
Be precise, academic, and critical.

Output EXACTLY these sections (in this order). Do not add, remove, or rename any section.

## Các Trường Phái Tư Tưởng
Identify 2-3 distinct positions in the literature. Use these labels:
- Trường phái Lạc Quan: [those who argue multi-agent clearly wins — cite evidence]
- Trường phái Hoài Nghi: [those who challenge or qualify the claim — cite evidence]
- Trường phái Có Điều Kiện: [those who say "it depends on the task type / scale"]

## Bằng Chứng Ủng Hộ Multi-Agent
For each piece of supporting evidence:
- [Tên paper/hệ thống] — [kết quả cụ thể, con số nếu có] — Độ tin cậy: [Cao/Trung/Thấp] — [lý do đánh giá]

## Bằng Chứng Phản Đối Multi-Agent
For each piece of challenging evidence (do NOT skip this section):
- [Tên paper/nguồn] — [lập luận cụ thể] — Độ tin cậy: [Cao/Trung/Thấp] — [lý do đánh giá]

## Phân Biệt: True Multi-Agent Gains vs. Confounders
This section is CRITICAL. Distinguish between:
- Token budget confound: Multi-agent systems often use 3-5x more tokens — is the comparison fair?
- Prompt engineering confound: Are gains from multi-agent architecture or from better prompts per agent?
- Self-reflection confound: Can a single-agent with repeated self-revision achieve the same result?
- Benchmark selection bias: Are the tasks chosen specifically suited to multi-agent decomposition?
Write 1-2 sentences per confounder, citing sources where possible.

## Bằng Chứng Thuyết Phục Là Gì?
Explicitly state: what experimental conditions would produce CONVINCING evidence for or against?
(e.g., "token-budget-matched comparison on standardized, task-diverse benchmark with blind human eval")

## Lo Ngại Phương Pháp Luận
List 3-5 specific methodological weaknesses found in the reviewed literature.
Format: [Loại lỗi] — [mô tả cụ thể] — [ảnh hưởng đến kết luận]

## Đề Xuất 3 Thí Nghiệm Cụ Thể
Each experiment MUST include: Hypothesis + Experimental design + Key metric + How confounders are controlled.
- Thí nghiệm 1 — [tên]:
  Hypothesis: ...
  Design: ...
  Metric: ...
  Confounder control: ...
- Thí nghiệm 2 — [tên]: [same format]
- Thí nghiệm 3 — [tên]: [same format]

## Gợi Ý Cho Writer
One sentence: the specific angle and tone the writer should adopt for the final judgment."""

_analyst_mod._USER_TEMPLATE = """\
Query: {query}
Target audience: {audience}

Research Notes:
{research_notes}

Analyse the notes above following the exact section structure in your instructions.
Be critical and concise. Each section bullet should be ≤ 2 sentences.
Total length: 350–550 words.

IMPORTANT: Respond entirely in Vietnamese (Tiếng Việt), including all section headers."""

# ─── Writer: xử lý Tasks 1, 8 và output format 7-section của Prompt 2 ────────
_writer_mod._SYSTEM_PROMPT = """\
You are helping a PhD student produce SPEAKING NOTES for a research group meeting.
Topic: "Do multi-agent LLM systems actually outperform single-agent systems on complex tasks?"

CRITICAL RULES — violating any of these will make the output useless:
1. Do NOT write a generic overview or intro like "This is an important topic..."
2. Every section must be concise and scannable — this is a PRESENTATION, not an essay.
3. Use bullet points within sections; keep each bullet to 1-2 sentences.
4. Cite all empirical claims inline as [1], [2], etc.
5. The "Kết Luận Cân Bằng" section MUST acknowledge genuine uncertainty and unresolved debates.
6. Do NOT end with a confident, closed conclusion — the field is genuinely unsettled.
7. The tone should be academic and balanced — present both sides fairly.

Output EXACTLY these 7 sections in this exact order. Do not add or merge sections.

## Câu Hỏi Cốt Lõi
Define the claim precisely in 2-3 sentences.
What exactly is being compared? Under what conditions? What does "outperform" mean?

## Các Luồng Ý Kiến Chính
3 bullets, one per school of thought (Lạc Quan / Hoài Nghi / Có Điều Kiện).
Each bullet: 1-2 sentences + representative evidence/source.

## Bằng Chứng Ủng Hộ
4-5 bullets. Format: [source/system] — [specific finding with numbers if available] — [credibility note]

## Bằng Chứng Phản Đối
4-5 bullets. Same format. This section is as important as "Bằng Chứng Ủng Hộ" — do not minimize it.

## Lo Ngại Phương Pháp Luận
5 bullets addressing specifically:
- Token budget fairness
- Prompt engineering confound
- Self-reflection confound
- Benchmark selection bias
- What WOULD count as convincing evidence

## Đề Xuất Thí Nghiệm
3 numbered experiments. Format per experiment:
**[Tên thí nghiệm]**: [Hypothesis] — [Key design choice] — [Primary metric]

## Kết Luận Cân Bằng
3-4 sentences MAXIMUM. Must include:
- What current evidence does support (with appropriate hedging)
- What remains unresolved or contested
- Explicit uncertainty: start the last sentence with "Câu hỏi này vẫn chưa có đồng thuận..."

Write entirely in Vietnamese (Tiếng Việt). Target total length: 500-650 words."""

_writer_mod._USER_TEMPLATE = """\
Query: {query}
Target audience: {audience}

Research Notes (từ Researcher Agent):
{research_notes}

Structured Analysis (từ Analyst Agent):
{analysis_notes}

Sources available (for citation):
{sources_list}

Write the final research briefing following the 7-section format in your instructions.
IMPORTANT: Respond entirely in Vietnamese (Tiếng Việt)."""

# =============================================================================
# BƯỚC 2 — Import sau khi đã patch prompts
# =============================================================================

from rich.console import Console          # noqa: E402
from rich.markdown import Markdown        # noqa: E402
from rich.panel import Panel              # noqa: E402

from multi_agent_research_lab.core.config import get_settings            # noqa: E402
from multi_agent_research_lab.core.schemas import ResearchQuery          # noqa: E402
from multi_agent_research_lab.core.state import ResearchState            # noqa: E402
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow   # noqa: E402
from multi_agent_research_lab.observability.logging import configure_logging  # noqa: E402
from multi_agent_research_lab.observability.tracing import (             # noqa: E402
    JSONFileTracer,
    configure_langsmith,
)

console = Console()

# =============================================================================
# BƯỚC 3 — Query Configuration (Dual-Search Strategy)
# =============================================================================

# Hai search queries để có đủ cả bằng chứng FOR và AGAINST:
# Query 1 tìm evidence PRO multi-agent (papers, benchmarks)
# Query 2 tìm evidence AGAINST/failures (methodological issues)
# → Merge thành 1 context query gửi cho agents
SEARCH_QUERY_PRO = (
    "AutoGen MetaGPT AgentBench multi-agent LLM outperform single-agent "
    "benchmark evaluation results 2023 2024"
)

SEARCH_QUERY_CON = (
    "multi-agent LLM systems fail limitations single-agent comparison "
    "token budget confounders empirical evaluation 2024"
)

# Context query đầy đủ gửi cho agents (không dùng để search)
CONTEXT_QUERY = (
    "Do multi-agent LLM systems actually outperform single-agent systems on complex tasks? "
    "Structured research briefing covering: major positions in literature, "
    "empirical evidence FOR and AGAINST (with named systems like AutoGen, MetaGPT, AgentBench), "
    "methodological concerns (token budget, prompt engineering, self-reflection confounders), "
    "3 concrete proposed experiments, balanced final judgment with uncertainty."
)

OUTPUT_DIR = Path("reports")
OUTPUT_FILE = OUTPUT_DIR / "demo_prompt2_result.md"


# =============================================================================
# BƯỚC 4 — Dual-Search Helper
# =============================================================================

def _fetch_merged_sources(max_per_query: int = 4) -> list:
    """Chạy 2 Tavily queries và merge kết quả để có đủ cả 2 phía."""
    from multi_agent_research_lab.services.search_client import SearchClient  # noqa: PLC0415
    searcher = SearchClient()

    console.print("[dim]Search 1/2: Tìm bằng chứng ủng hộ multi-agent...[/dim]")
    try:
        sources_pro = searcher.search(SEARCH_QUERY_PRO, max_results=max_per_query)
    except Exception:
        sources_pro = []

    console.print("[dim]Search 2/2: Tìm bằng chứng phản đối / thất bại...[/dim]")
    try:
        sources_con = searcher.search(SEARCH_QUERY_CON, max_results=max_per_query)
    except Exception:
        sources_con = []

    # Deduplicate by URL
    seen_urls: set[str] = set()
    merged: list = []
    for src in sources_pro + sources_con:
        url = src.url or src.title
        if url not in seen_urls:
            seen_urls.add(url)
            merged.append(src)

    console.print(f"[dim]Merged: {len(sources_pro)} pro + {len(sources_con)} con = {len(merged)} unique sources[/dim]")
    return merged


# =============================================================================
# BƯỚC 5 — Chạy Demo
# =============================================================================

def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_langsmith(settings.langsmith_api_key, settings.langsmith_project)

    console.print(
        Panel.fit(
            "[bold magenta]Demo: Prompt 2 — Research Briefing on Multi-Agent LLMs[/bold magenta]\n"
            "[dim]Showcase demo — 5 điểm bonus[/dim]",
            style="magenta",
        )
    )
    console.print(f"\n[dim]Context query:[/dim] {CONTEXT_QUERY[:100]}...\n")

    # ── Dual-search để có đủ nguồn cả 2 phía ────────────────────────────────
    merged_sources = _fetch_merged_sources(max_per_query=4)

    # ── Tạo state với audience = PhD student ─────────────────────────────────
    request = ResearchQuery(
        query=CONTEXT_QUERY,
        audience="PhD student và research group",
        max_sources=len(merged_sources) or 7,
    )
    state = ResearchState(request=request)
    # Pre-populate sources để Researcher không search lại từ đầu
    state.sources = merged_sources

    # ── Patch Researcher để dùng sources có sẵn thay vì search lại ──────────
    # Override _search.search để trả về merged_sources ngay lập tức
    from multi_agent_research_lab.services.search_client import SearchClient  # noqa: PLC0415
    class _PreloadedSearch(SearchClient):
        def search(self, query: str, max_results: int = 5) -> list:  # type: ignore[override]
            return merged_sources

    from multi_agent_research_lab.agents.researcher import ResearcherAgent  # noqa: PLC0415
    _original_init = ResearcherAgent.__init__

    def _patched_init(self_agent, llm=None, search=None):  # type: ignore[misc]
        _original_init(self_agent, llm=llm, search=_PreloadedSearch())

    ResearcherAgent.__init__ = _patched_init  # type: ignore[method-assign]

    # ── Chạy workflow ─────────────────────────────────────────────────────────
    console.print("[cyan]Đang chạy Multi-Agent workflow...[/cyan]")
    started = time.perf_counter()

    workflow = MultiAgentWorkflow()
    final_state = workflow.run(state)

    elapsed = time.perf_counter() - started

    # ── Route history ─────────────────────────────────────────────────────────
    if final_state.route_history:
        route_str = " → ".join(final_state.route_history)
        console.print(f"\n[dim]Route: {route_str}[/dim]")

    if final_state.errors:
        for err in final_state.errors:
            console.print(f"[yellow]⚠ {err}[/yellow]")

    # ── Section verification ──────────────────────────────────────────────────
    answer = final_state.final_answer or ""
    _verify_sections(answer)

    # ── In output ─────────────────────────────────────────────────────────────
    if answer:
        console.print("\n")
        console.print(Markdown(answer))

    # ── Stats ─────────────────────────────────────────────────────────────────
    console.print(
        Panel.fit(
            f"Sources: {len(final_state.sources)}  |  "
            f"Tokens: {final_state.total_tokens}  |  "
            f"Cost: ${final_state.total_cost_usd:.6f}  |  "
            f"Citations: {final_state.citation_count}  |  "
            f"Elapsed: {elapsed:.2f}s",
            title="Stats",
            style="green",
        )
    )

    # ── Lưu output ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_result(final_state, elapsed)

    # ── Lưu trace ─────────────────────────────────────────────────────────────
    if final_state.trace:
        tracer = JSONFileTracer()
        trace_path = tracer.save(
            trace_events=final_state.trace,
            metadata={
                "demo": "prompt2",
                "context_query": CONTEXT_QUERY,
                "search_query_pro": SEARCH_QUERY_PRO,
                "search_query_con": SEARCH_QUERY_CON,
                "route_history": final_state.route_history,
                "total_tokens": final_state.total_tokens,
                "total_cost_usd": final_state.total_cost_usd,
                "elapsed_seconds": elapsed,
                "errors": final_state.errors,
                "citation_count": final_state.citation_count,
                "source_count": len(final_state.sources),
            },
        )
        console.print(f"\n[dim]JSON trace → {trace_path}[/dim]")

    console.print(f"\n[green]✓ Kết quả đã lưu → {OUTPUT_FILE}[/green]")
    console.print(
        "[dim]Nộp thầy giáo: reports/demo_prompt2_result.md + link LangSmith[/dim]"
    )


def _verify_sections(answer: str) -> None:
    """Kiểm tra output có đủ 7 sections của Prompt 2 không."""
    required = [
        "Câu Hỏi Cốt Lõi",
        "Luồng Ý Kiến",
        "Bằng Chứng Ủng Hộ",
        "Bằng Chứng Phản Đối",
        "Lo Ngại Phương Pháp",
        "Đề Xuất Thí Nghiệm",
        "Kết Luận",
    ]
    console.print("\n[bold]Kiểm tra đủ 7 sections:[/bold]")
    all_ok = True
    for section in required:
        found = section.lower() in answer.lower()
        status = "✅" if found else "❌ THIẾU"
        console.print(f"  {status}  {section}")
        if not found:
            all_ok = False
    if not all_ok:
        console.print(
            "[yellow]⚠ Một số section bị thiếu. Xem lại output và chạy lại nếu cần.[/yellow]"
        )
    else:
        console.print("[green]✅ Đủ tất cả 7 sections — sẵn sàng nộp![/green]")


def _save_result(state: ResearchState, elapsed: float) -> None:
    """Lưu kết quả ra reports/demo_prompt2_result.md."""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sources_section = ""
    if state.sources:
        lines = [
            f"[{i}] [{s.title}]({s.url or '#'})"
            for i, s in enumerate(state.sources, 1)
        ]
        sources_section = "\n---\n\n## Nguồn Tham Khảo\n\n" + "\n".join(lines)

    content = f"""# Báo Cáo Nghiên Cứu — Demo Prompt 2
## Hệ Thống Multi-Agent Research

**Tác giả:** Nguyễn Đức Mạnh — 2A202600945  
**Thời gian:** {now}  
**Route:** {' → '.join(state.route_history)}  

---

**Câu hỏi gốc (Prompt 2):**
> "Do multi-agent LLM systems actually outperform single-agent systems on complex tasks?"

---

{state.final_answer or '[Không có output]'}
{sources_section}

---

## Thống Kê Chạy

| Metric | Giá trị |
|---|---|
| Độ trễ (Latency) | {elapsed:.2f}s |
| Tổng tokens | {state.total_tokens:,} |
| Chi phí ước tính | ${state.total_cost_usd:.6f} |
| Số trích dẫn | {state.citation_count} |
| Số nguồn web | {len(state.sources)} |
| Số bước agent | {state.iteration} |
| Lỗi xảy ra | {len(state.errors)} |
"""
    OUTPUT_FILE.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
