"""Search client abstraction for ResearcherAgent.

Backed by Tavily for real web search. Falls back gracefully if key is missing.
"""

from __future__ import annotations

import logging
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Tavily-backed search client with retry logic."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: Any = None  # lazy-init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from tavily import TavilyClient  # noqa: PLC0415
            except ImportError as exc:
                raise AgentExecutionError(
                    "tavily-python package not installed. Run: pip install tavily-python"
                ) from exc

            if not self._settings.tavily_api_key:
                raise AgentExecutionError(
                    "TAVILY_API_KEY is not set. Check your .env file."
                )

            self._client = TavilyClient(api_key=self._settings.tavily_api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search the web via Tavily and return structured SourceDocuments.

        Each basic search call costs 1 Tavily credit (1,000 free/month).
        """
        client = self._get_client()

        logger.debug("SearchClient.search | query='%s' | max_results=%d", query, max_results)

        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",  # basic=1 credit, advanced=2 credits
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)
            raise AgentExecutionError(f"Search failed: {exc}") from exc

        results = response.get("results", [])
        logger.debug("SearchClient.search | returned %d results", len(results))

        documents: list[SourceDocument] = []
        for item in results:
            documents.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={
                        "score": item.get("score"),
                        "published_date": item.get("published_date"),
                    },
                )
            )

        return documents
