"""Reproduce the customer-reported latency issue with web_search=True.

Simulates a research-style query that triggers multiple searches. Instruments
all API calls to show where time is spent and how context is (not) accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

from thetokencompany._types import CompressResponse, SearchResponse, SearchResult

# ---------------------------------------------------------------------------
# Mock response objects (same as test_web_search.py)
# ---------------------------------------------------------------------------

@dataclass
class _ContentBlock:
    type: str
    text: str | None = None
    name: str | None = None
    id: str | None = None
    input: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        if self.name is not None:
            d["name"] = self.name
        if self.id is not None:
            d["id"] = self.id
        if self.input is not None:
            d["input"] = self.input
        return d


@dataclass
class _MockResponse:
    stop_reason: str
    content: list[_ContentBlock]


def _search_response(query: str, tool_id: str) -> _MockResponse:
    return _MockResponse(
        stop_reason="tool_use",
        content=[
            _ContentBlock(
                type="tool_use",
                name="ttc_web_search",
                id=tool_id,
                input={"query": query},
            ),
        ],
    )


def _text_response(text: str) -> _MockResponse:
    return _MockResponse(
        stop_reason="end_turn",
        content=[_ContentBlock(type="text", text=text)],
    )


def _make_search_result(query: str) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResult(
                url=f"https://example.com/{query.replace(' ', '-')}",
                title=f"Results for: {query}",
                content=f"Detailed article content about {query}. " * 50,
                score=0.9,
            ),
            SearchResult(
                url=f"https://news.example.com/{query.replace(' ', '-')}",
                title=f"News: {query}",
                content=f"Breaking news about {query}. " * 30,
                score=0.85,
            ),
        ],
        query=query,
        original_input_tokens=2000,
        output_tokens=800,
    )


# ---------------------------------------------------------------------------
# Simulate the customer's research query
# ---------------------------------------------------------------------------

SEARCH_QUERIES = [
    "AI content creation market leaders Jasper Copy.ai Writer.com 2026",
    "enterprise AI content creation adoption trends 2026",
    "AI content creation ROI metrics case studies",
    "AI asset performance intelligence tools",
    "AI content creation competitive landscape analysis",
]


def test_repro_research_query_latency():
    """Reproduce: research query triggers multiple searches, context is lost.

    The model would keep searching because it loses prior results each
    iteration. We simulate 5 search iterations + final text response.
    """
    from thetokencompany.anthropic import with_compression

    mock_client = MagicMock()

    # Model searches 5 times then responds with text
    responses = [
        _search_response(q, f"toolu_{i}") for i, q in enumerate(SEARCH_QUERIES)
    ]
    responses.append(_text_response("Here is the competitive analysis..."))

    original_create = MagicMock(side_effect=responses)
    mock_client.messages.create = original_create

    # Track all API calls
    compress_call_log: list[dict[str, Any]] = []
    search_call_log: list[dict[str, Any]] = []

    def mock_compress(text: str, *, model: str, aggressiveness: float) -> CompressResponse:
        compress_call_log.append({
            "text_len": len(text),
            "text_preview": text[:80],
        })
        # Simulate ~100ms network latency per compress call
        return CompressResponse(output=text, output_tokens=10, input_tokens=10)

    def mock_search(query: str) -> SearchResponse:
        search_call_log.append({"query": query})
        # Simulate ~2s for web search + compression
        return _make_search_result(query)

    with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
        mock_ttc = MockTTC.return_value
        mock_ttc.compress.side_effect = mock_compress
        mock_ttc.search.side_effect = mock_search

        wrapped = with_compression(
            mock_client,
            compression_api_key="ttc-test",
            web_search=True,
        )

        user_query = (
            "Research competitive intelligence for: AI-Powered Content Creation "
            "and Asset Performance Intelligence Focus on: 1. Market leaders in "
            "AI content creation (Jasper, Copy.ai, Writer.com) 2. Enterprise "
            "adoption trends 3. ROI metrics and case studies Provide a "
            "structured analysis with sources."
        )

        wrapped.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system="You are a research analyst. Provide thorough analysis.",
            messages=[{"role": "user", "content": user_query}],
        )

    # --- Diagnostics ---
    print("\n" + "=" * 70)
    print("REPRODUCTION: Research query with web_search=True")
    print("=" * 70)

    print(f"\nLLM API calls (original_create): {original_create.call_count}")
    print(f"TTC compress API calls:          {len(compress_call_log)}")
    print(f"TTC search API calls:            {len(search_call_log)}")

    # Check context accumulation
    print("\n--- Context in each LLM call ---")
    for i, call in enumerate(original_create.call_args_list):
        kwargs = call[1]
        msgs = kwargs.get("messages", [])
        assistant_count = sum(1 for m in msgs if m["role"] == "assistant")
        tool_result_count = sum(
            1
            for m in msgs
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"]
            )
        )
        print(
            f"  Call {i + 1}: {len(msgs)} messages "
            f"(assistant: {assistant_count}, tool_results: {tool_result_count})"
        )

    # Check for re-compression of already-compressed content
    print("\n--- Compress API call details ---")
    for i, c in enumerate(compress_call_log):
        print(f"  #{i + 1}: len={c['text_len']} | {c['text_preview']}...")

    # Estimate real-world latency
    # Each compress call: ~300ms (network + processing)
    # Each search call: ~2-5s (web search + compression)
    # Each LLM call: ~3-10s
    compress_latency_ms = len(compress_call_log) * 300
    search_latency_ms = len(search_call_log) * 3000
    llm_latency_ms = original_create.call_count * 5000
    total_est_ms = compress_latency_ms + search_latency_ms + llm_latency_ms

    print("\n--- Estimated real-world latency ---")
    print(f"  Compress calls: {len(compress_call_log)} × ~300ms = ~{compress_latency_ms}ms")
    print(f"  Search calls:   {len(search_call_log)} × ~3s    = ~{search_latency_ms}ms")
    print(f"  LLM calls:      {original_create.call_count} × ~5s    = ~{llm_latency_ms}ms")
    print(f"  TOTAL ESTIMATE: ~{total_est_ms / 1000:.0f}s ({total_est_ms / 60000:.1f} min)")

    print("\n--- Compression stats ---")
    print(f"  total_tokens_saved: {wrapped.compression.total_tokens_saved}")
    print(f"  turns recorded: {wrapped.compression.calls}")

    # Assert the bugs
    print("\n--- Bug verification ---")

    # Bug 1: Context loss
    last_call_kwargs = original_create.call_args_list[-1][1]
    last_msgs = last_call_kwargs["messages"]
    assistant_msgs = [m for m in last_msgs if m["role"] == "assistant"]
    expected_assistants = len(SEARCH_QUERIES)
    print(
        f"  Context loss: final call has {len(assistant_msgs)} assistant msgs "
        f"(expected {expected_assistants})"
    )
    assert len(assistant_msgs) == expected_assistants, (
        f"BUG CONFIRMED: Context loss — final LLM call has {len(assistant_msgs)} "
        f"assistant messages instead of {expected_assistants}. Previous search "
        f"results are being dropped each iteration."
    )
