"""Benchmark report rendering."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from multi_agent_research_lab.core.schemas import BenchmarkMetrics

logger = logging.getLogger(__name__)

_REPORT_DIR = Path("reports")


def render_markdown_report(metrics: list[BenchmarkMetrics], query: str = "") -> str:
    """Render benchmark metrics to a rich Markdown report (Vietnamese).

    Includes:
    - Summary table (Latency, Cost, Quality, Notes)
    - Analysis section
    - Failure mode discussion
    - Exit ticket answers
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = [
        "# Báo Cáo Benchmark — Hệ Thống Multi-Agent Research",
        "",
        f"**Thời gian tạo:** {now}  ",
        f"**Câu hỏi:** {query}  " if query else "",
        "",
        "---",
        "",
        "## Kết Quả",
        "",
        "| Chế độ chạy | Độ trễ (s) | Chi phí (USD) | Chất lượng (0–10) | Ghi chú |",
        "|---|---:|---:|:---:|---|",
    ]

    for item in metrics:
        cost = "N/A" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.6f}"
        quality = "N/A" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Phân Tích So Sánh",
        "",
    ]

    # Auto-generate comparison if we have exactly 2 runs
    if len(metrics) == 2:
        single = next((m for m in metrics if "single" in m.run_name.lower()), metrics[0])
        multi  = next((m for m in metrics if "multi"  in m.run_name.lower()), metrics[1])

        lat_diff = multi.latency_seconds - single.latency_seconds
        lat_pct  = (lat_diff / single.latency_seconds * 100) if single.latency_seconds else 0
        lat_label = "chậm hơn" if lat_diff > 0 else "nhanh hơn"

        cost_diff = (multi.estimated_cost_usd or 0) - (single.estimated_cost_usd or 0)
        cost_label = "đắt hơn" if cost_diff > 0 else "rẻ hơn"

        q_single = single.quality_score or 0
        q_multi  = multi.quality_score or 0
        q_diff   = q_multi - q_single

        lines += [
            f"- **Độ trễ**: Multi-agent **{abs(lat_diff):.2f}s {lat_label}** "
            f"({lat_pct:+.1f}% so với single-agent).",
            f"- **Chi phí**: Multi-agent **{cost_label} ${abs(cost_diff):.6f}** mỗi câu hỏi.",
            f"- **Chất lượng**: Multi-agent đạt **{q_diff:+.1f} điểm** so với single-agent.",
            "",
            "### Khi nào nên dùng multi-agent?",
            "- Câu hỏi nghiên cứu phức tạp, cần tổng hợp nhiều nguồn thông tin.",
            "- Tác vụ có thể tách rõ: tìm kiếm → phân tích → viết (chuyên môn hoá).",
            "- Workflow cần audit trail rõ ràng cho từng bước của agent.",
            "",
            "### Khi nào KHÔNG nên dùng multi-agent?",
            "- Câu hỏi đơn giản, có thể trả lời bằng một LLM call duy nhất.",
            "- Ứng dụng yêu cầu độ trễ thấp — nhiều vòng API call là không chấp nhận được.",
            "- Ngân sách hạn chế — chi phí tăng 2–3× so với single-agent.",
            "",
        ]
    else:
        lines += ["*(Thêm phân tích so sánh ở đây)*", ""]

    lines += [
        "---",
        "",
        "## Chi Tiết Câu Trả Lời",
        "",
    ]

    for item in metrics:
        if "single" in item.run_name.lower():
            display_name = "Single-Agent (Baseline)"
        elif "multi" in item.run_name.lower():
            display_name = "Multi-Agent (Workflow)"
        else:
            display_name = item.run_name.title()

        content = item.response_content or "*(Không có câu trả lời)*"
        lines += [
            f"### {display_name}",
            "",
            content,
            "",
        ]

    lines += [
        "---",
        "",
        "## Phân Tích Điểm Thất Bại (Failure Mode)",
        "",
        "| Lỗi | Nguyên nhân | Cách xử lý đã implement |",
        "|---|---|---|",
        "| Tavily tìm không ra kết quả | Query quá hẹp / API lỗi | Fallback sang ghi chú từ kiến thức mô hình |",
        "| LLM timeout / rate limit | Provider quá tải | tenacity retry (3 lần, exponential backoff) |",
        "| Workflow lặp vô hạn | Lỗi logic Supervisor | Guardrail `max_iterations` → ép về `done` |",
        "| `final_answer` rỗng | Writer không có notes đầu vào | Guard check trong `WriterAgent.run()` |",
        "",
        "---",
        "",
        "## Exit Ticket",
        "",
        "**Câu 1: Khi nào nên dùng multi-agent?**",
        "",
        "> Nên dùng multi-agent khi tác vụ có thể tách thành các bước con độc lập và được hưởng lợi "
        "từ sự chuyên môn hoá (ví dụ: tìm kiếm → phân tích → viết). Chi phí và độ trễ tăng thêm "
        "được chấp nhận khi chất lượng cải thiện đáng kể và tác vụ đủ phức tạp để một context window "
        "đơn lẻ không thể xử lý đáng tin cậy.",
        "",
        "**Câu 2: Khi nào KHÔNG nên dùng multi-agent?**",
        "",
        "> Tránh dùng multi-agent cho những câu hỏi đơn giản, cần phản hồi nhanh, mà một LLM call "
        "được prompt tốt là đủ. Chi phí điều phối (nhiều API call, tuần tự hoá trạng thái) có thể "
        "tốn hơn 2–3× và mất thời gian hơn 3× mà chất lượng không cải thiện tương xứng.",
        "",
    ]

    return "\n".join(line for line in lines if line is not None)


def save_report(metrics: list[BenchmarkMetrics], query: str = "") -> Path:
    """Write the markdown report to reports/benchmark_report.md."""
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORT_DIR / "benchmark_report.md"
    content = render_markdown_report(metrics, query=query)
    path.write_text(content, encoding="utf-8")
    logger.info("Benchmark report saved to %s", path)
    return path
