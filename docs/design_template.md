# Design Template — Multi-Agent Research System

**Tác giả:** Nguyễn Đức Mạnh — 2A202600945  
**Ngày:** 2026-06-24

---

## Problem

Xây dựng một **research assistant** có thể nhận câu hỏi phức tạp từ người dùng, tự động tìm kiếm thông tin từ web, phân tích và tổng hợp thành câu trả lời ~500 chữ có trích dẫn nguồn.

**Task cụ thể:** Nhận query dạng "Research X and write a summary" → trả về bài viết có cấu trúc rõ ràng, có source citation, phù hợp với đối tượng kỹ thuật.

---

## Why multi-agent?

Single-agent **không đủ** vì:

1. **Context limit**: Một agent phải làm search + analysis + writing trong một prompt → context bị nén, chất lượng giảm.
2. **Thiếu chuyên môn hoá**: LLM giỏi hơn khi được giao một task rõ ràng (chỉ search, hoặc chỉ phân tích).
3. **Khó debug**: Nếu output tệ, không biết lỗi ở bước nào (tìm sai? phân tích sai? viết sai?).
4. **Không có trace**: Single-agent không cho biết từng bước làm gì, tốn bao nhiêu token.

Multi-agent giải quyết bằng cách tách rõ 4 roles với state rõ ràng giữa các bước.

---

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| **Supervisor** | Quyết định agent tiếp theo; enforce guardrails | `ResearchState` (tất cả fields) | Cập nhật `route_history` | Max iterations → force `done` |
| **Researcher** | Tìm kiếm web (Tavily) + tóm tắt nguồn | `request.query`, `request.max_sources` | `state.sources`, `state.research_notes` | Search fail → fallback to empty note; LLM fail → placeholder |
| **Analyst** | Phân tích research notes → trích claim, viewpoints, gaps | `state.research_notes`, `request.query` | `state.analysis_notes` (structured) | No research notes → skip with warning |
| **Writer** | Tổng hợp → viết final answer có citation | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` | Missing notes → guard + error log |

---

## Shared state (`ResearchState`)

| Field | Type | Lý do cần |
|---|---|---|
| `request` | `ResearchQuery` | Query gốc + config (max_sources, audience) |
| `iteration` | `int` | Đếm bước, dùng cho guardrail max_iterations |
| `route_history` | `list[str]` | Audit trail — ai đã chạy theo thứ tự nào |
| `sources` | `list[SourceDocument]` | Kết quả Tavily — Writer dùng để build citation list |
| `research_notes` | `str \| None` | Output của Researcher → input của Analyst & Writer |
| `analysis_notes` | `str \| None` | Output của Analyst → input của Writer |
| `final_answer` | `str \| None` | Output cuối cùng trả cho user |
| `agent_results` | `list[AgentResult]` | Per-agent metadata (tokens, cost) |
| `trace` | `list[dict]` | Structured events với timestamp |
| `errors` | `list[str]` | Non-fatal errors — không làm crash workflow |
| `total_input_tokens` | `int` | Benchmark: tổng input tokens |
| `total_output_tokens` | `int` | Benchmark: tổng output tokens |
| `total_cost_usd` | `float` | Benchmark: tổng chi phí |
| `start_time` | `float \| None` | Tính elapsed time |

---

## Routing policy

```
START → Supervisor
         │
         ├── research_notes is None?  ──→  Researcher ──→ Supervisor
         │
         ├── analysis_notes is None?  ──→  Analyst    ──→ Supervisor
         │
         ├── final_answer is None?    ──→  Writer     ──→ Supervisor
         │
         └── all populated?           ──→  END
         
         (GUARDRAIL: iteration >= max_iterations → force END)
```

**Chiến lược routing**: Rule-based (không tốn LLM call) → deterministic, fast, dễ debug.

---

## Guardrails

| Guardrail | Giá trị | Cách implement |
|---|---|---|
| **Max iterations** | 6 (config) | Supervisor check `iteration >= max_iterations` → route `done` |
| **Timeout** | 60s (config) | OpenAI client `timeout=settings.timeout_seconds` |
| **Retry** | 3 lần, exp. backoff | `tenacity.retry(stop=stop_after_attempt(3), wait=wait_exponential())` |
| **Fallback** | Mỗi agent có fallback | Search fail → empty sources; LLM fail → placeholder text |
| **Validation** | Pydantic schemas | `ResearchQuery`, `SourceDocument`, `BenchmarkMetrics` đều dùng Pydantic v2 |

---

## Benchmark plan

**Queries:**
1. `"What is GraphRAG and how does it improve over standard RAG?"`
2. `"Explain the differences between LangGraph and CrewAI for multi-agent systems"`
3. `"What are the latest advances in LLM reasoning and chain-of-thought prompting?"`

**Metrics:**

| Metric | Cách đo |
|---|---|
| Latency (s) | `perf_counter()` wall-clock time |
| Cost (USD) | Token count × price/token (gpt-4o-mini) |
| Quality (0–10) | Heuristic: độ dài (5pts) + có notes (2pts) + citations (3pts) |
| Citation count | Đếm `[N]` patterns trong final_answer |
| Failure rate | `len(state.errors) > 0` |

**Expected outcome:**
- Multi-agent: latency 3–5× cao hơn, cost 2–3× cao hơn, quality 1–3 điểm cao hơn.
- Single-agent: nhanh và rẻ hơn nhưng thiếu citation và depth.
