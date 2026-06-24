"""Tests for web_search support in the Anthropic with_compression wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thetokencompany._types import (
    ChatCompressResponse,
    CompressResponse,
    SearchResponse,
    SearchResult,
)
from thetokencompany.anthropic import (
    _format_search_results,
    _has_ttc_search_use,
    _inject_search_tool,
)


def _mock_compress_response(text: str) -> CompressResponse:
    return CompressResponse(
        output=f"[compressed]{text}",
        output_tokens=5,
        input_tokens=20,
    )


def _chat_echo(messages: list, **kwargs: object) -> ChatCompressResponse:
    """Echo messages back (server-side compression is mocked away here)."""
    return ChatCompressResponse(
        messages=messages,
        system=kwargs.get("system"),
        input_tokens=20,
        output_tokens=5,
        cache_hits=0,
        cache_misses=len(messages),
        compression_time=0.0,
    )


def _chat_noop(messages: list, **kwargs: object) -> ChatCompressResponse:
    """Echo with zero token savings, to isolate web-search savings in stats."""
    return ChatCompressResponse(
        messages=messages,
        system=kwargs.get("system"),
        input_tokens=10,
        output_tokens=10,
        cache_hits=0,
        cache_misses=len(messages),
        compression_time=0.0,
    )


def _make_search_response(
    query: str = "test",
    results: list[SearchResult] | None = None,
) -> SearchResponse:
    if results is None:
        results = [
            SearchResult(
                url="https://example.com",
                title="Example",
                content="Example content about the query.",
                score=0.95,
            ),
        ]
    return SearchResponse(
        results=results,
        query=query,
        original_input_tokens=1000,
        output_tokens=700,
    )


# ---------------------------------------------------------------------------
# Mock Anthropic response objects
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


def _text_response(text: str = "Hello!") -> _MockResponse:
    return _MockResponse(
        stop_reason="end_turn",
        content=[_ContentBlock(type="text", text=text)],
    )


def _search_tool_use_response(
    query: str = "latest news",
    tool_use_id: str = "toolu_abc123",
) -> _MockResponse:
    return _MockResponse(
        stop_reason="tool_use",
        content=[
            _ContentBlock(
                type="tool_use",
                name="ttc_web_search",
                id=tool_use_id,
                input={"query": query},
            ),
        ],
    )


def _mixed_tool_use_response(
    search_query: str = "latest news",
    other_tool_name: str = "calculator",
) -> _MockResponse:
    return _MockResponse(
        stop_reason="tool_use",
        content=[
            _ContentBlock(
                type="tool_use",
                name="ttc_web_search",
                id="toolu_search1",
                input={"query": search_query},
            ),
            _ContentBlock(
                type="tool_use",
                name=other_tool_name,
                id="toolu_other1",
                input={"expression": "2+2"},
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestInjectSearchTool:
    def test_adds_tool_when_empty(self) -> None:
        kwargs: dict[str, Any] = {}
        _inject_search_tool(kwargs)
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["name"] == "ttc_web_search"

    def test_removes_server_side_web_search(self) -> None:
        kwargs: dict[str, Any] = {
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "calculator", "input_schema": {}},
            ],
        }
        _inject_search_tool(kwargs)
        names = [t.get("name") for t in kwargs["tools"]]
        assert "ttc_web_search" in names
        types = [t.get("type") for t in kwargs["tools"]]
        assert "web_search_20250305" not in types

    def test_removes_any_web_search_version(self) -> None:
        kwargs: dict[str, Any] = {
            "tools": [
                {"type": "web_search_20260209", "name": "web_search"},
            ],
        }
        _inject_search_tool(kwargs)
        types = [t.get("type") for t in kwargs["tools"]]
        assert not any(str(t).startswith("web_search_") for t in types)
        assert kwargs["tools"][0]["name"] == "ttc_web_search"

    def test_preserves_existing_tools(self) -> None:
        kwargs: dict[str, Any] = {
            "tools": [{"name": "calculator", "input_schema": {}}],
        }
        _inject_search_tool(kwargs)
        names = [t["name"] for t in kwargs["tools"]]
        assert "calculator" in names
        assert "ttc_web_search" in names
        assert len(kwargs["tools"]) == 2

    def test_does_not_duplicate(self) -> None:
        kwargs: dict[str, Any] = {
            "tools": [
                {"name": "ttc_web_search", "input_schema": {}},
            ],
        }
        _inject_search_tool(kwargs)
        ttc_tools = [t for t in kwargs["tools"] if t.get("name") == "ttc_web_search"]
        assert len(ttc_tools) == 1


class TestHasTtcSearchUse:
    def test_no_tool_use_stop_reason(self) -> None:
        assert not _has_ttc_search_use(_text_response())

    def test_ttc_search_present(self) -> None:
        assert _has_ttc_search_use(_search_tool_use_response())

    def test_other_tool_only(self) -> None:
        resp = _MockResponse(
            stop_reason="tool_use",
            content=[
                _ContentBlock(
                    type="tool_use",
                    name="calculator",
                    id="toolu_1",
                    input={},
                ),
            ],
        )
        assert not _has_ttc_search_use(resp)

    def test_mixed_tools(self) -> None:
        assert _has_ttc_search_use(_mixed_tool_use_response())


class TestFormatSearchResults:
    def test_formats_correctly(self) -> None:
        sr = _make_search_response()
        text = _format_search_results(sr)
        assert "Source: Example" in text
        assert "URL: https://example.com" in text
        assert "Example content about the query." in text

    def test_empty_results(self) -> None:
        sr = _make_search_response(results=[])
        text = _format_search_results(sr)
        assert text == ""

    def test_multiple_results(self) -> None:
        sr = _make_search_response(
            results=[
                SearchResult(
                    url="https://a.com",
                    title="A",
                    content="Content A",
                ),
                SearchResult(
                    url="https://b.com",
                    title="B",
                    content="Content B",
                ),
            ],
        )
        text = _format_search_results(sr)
        assert "Source: A" in text
        assert "Source: B" in text


# ---------------------------------------------------------------------------
# Integration tests for sync wrapper
# ---------------------------------------------------------------------------


class TestWebSearchSyncWrapper:
    def test_web_search_false_no_change(self) -> None:
        """web_search=False should not inject tools or modify behavior."""
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value=_text_response())
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress.return_value = _mock_compress_response("x")

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=False,
            )
            result = wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"name": "calculator", "input_schema": {}}],
            )

        call_kwargs = original_create.call_args[1]
        tool_names = [t.get("name") for t in call_kwargs["tools"]]
        assert "ttc_web_search" not in tool_names
        assert result.stop_reason == "end_turn"

    def test_web_search_true_injects_tool(self) -> None:
        """web_search=True should add ttc_web_search to tools."""
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value=_text_response())
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress.return_value = _mock_compress_response("x")

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
            )

        call_kwargs = original_create.call_args[1]
        tool_names = [t.get("name") for t in call_kwargs["tools"]]
        assert "ttc_web_search" in tool_names

    def test_web_search_true_removes_server_tool(self) -> None:
        """web_search=True should remove web_search_20250305."""
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value=_text_response())
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress.return_value = _mock_compress_response("x")

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                tools=[
                    {"type": "web_search_20250305", "name": "web_search"},
                ],
            )

        call_kwargs = original_create.call_args[1]
        types = [t.get("type") for t in call_kwargs["tools"]]
        assert "web_search_20250305" not in types

    def test_search_loop_handles_tool_use(self) -> None:
        """The wrapper should call TTC search and loop back."""
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()

        # First call returns tool_use, second returns text
        original_create = MagicMock(
            side_effect=[
                _search_tool_use_response("latest news"),
                _text_response("Here is the news."),
            ],
        )
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress.return_value = _mock_compress_response("x")
            mock_ttc.search.return_value = _make_search_response(
                query="latest news",
            )

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            result = wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is the news?"}],
            )

        # Should have called create twice
        assert original_create.call_count == 2

        # Second call should have the tool_result in messages
        second_call_kwargs = original_create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        # Last message should be user with tool_result
        last_msg = messages[-1]
        assert last_msg["role"] == "user"
        assert last_msg["content"][0]["type"] == "tool_result"
        assert last_msg["content"][0]["tool_use_id"] == "toolu_abc123"

        # Final result should be the text response
        assert result.stop_reason == "end_turn"
        assert result.content[0].text == "Here is the news."

        # TTC search should have been called
        mock_ttc.search.assert_called_once_with("latest news")

    def test_search_loop_stops_on_other_tool(self) -> None:
        """If only non-ttc tools remain, return to user."""
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()

        # First call: mixed tool use (ours + other)
        # Second call: only other tool use (no ttc_web_search)
        other_tool_response = _MockResponse(
            stop_reason="tool_use",
            content=[
                _ContentBlock(
                    type="tool_use",
                    name="calculator",
                    id="toolu_calc1",
                    input={"expression": "2+2"},
                ),
            ],
        )
        original_create = MagicMock(
            side_effect=[
                _mixed_tool_use_response(),
                other_tool_response,
            ],
        )
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress.return_value = _mock_compress_response("x")
            mock_ttc.search.return_value = _make_search_response()

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            result = wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Search and calc"}],
            )

        # Should return the response with only the calculator tool
        assert result.stop_reason == "tool_use"
        assert result.content[0].name == "calculator"
        assert original_create.call_count == 2

    def test_no_loop_when_no_search_in_response(self) -> None:
        """If response has no ttc_web_search, no loop occurs."""
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value=_text_response())
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress.return_value = _mock_compress_response("x")

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            result = wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
            )

        assert original_create.call_count == 1
        assert result.stop_reason == "end_turn"
        mock_ttc.search.assert_not_called()


    def test_multi_search_preserves_context(self) -> None:
        """Multiple search iterations must accumulate context, not lose it.

        Bug: the search loop reads from the original kwargs each iteration,
        so previous assistant+tool_result pairs are dropped. This causes the
        model to lose context and potentially re-search indefinitely.
        """
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()

        # Simulate 3 calls: search1 → search2 → final text
        original_create = MagicMock(
            side_effect=[
                _search_tool_use_response("query 1", "toolu_1"),
                _search_tool_use_response("query 2", "toolu_2"),
                _text_response("Final answer."),
            ],
        )
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress.return_value = _mock_compress_response("x")
            mock_ttc.search.return_value = _make_search_response()

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Research AI"}],
            )

        assert original_create.call_count == 3

        # The third call (after 2 searches) must contain ALL prior context:
        # original user msg + assistant1 + tool_result1 + assistant2 + tool_result2
        third_call_kwargs = original_create.call_args_list[2][1]
        messages = third_call_kwargs["messages"]

        # Count assistant and user messages with tool_results
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        tool_result_msgs = [
            m
            for m in messages
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in m["content"]
            )
        ]

        assert len(assistant_msgs) == 2, (
            f"Expected 2 assistant messages (both search responses), got {len(assistant_msgs)}. "
            "Context from earlier search iterations is being lost."
        )
        assert len(tool_result_msgs) == 2, (
            f"Expected 2 tool_result messages (both search results), got {len(tool_result_msgs)}. "
            "Context from earlier search iterations is being lost."
        )

    def test_search_loop_does_not_recompress_already_compressed(self) -> None:
        """Messages compressed in the initial call should not be re-compressed
        in the search loop, since they're already compressed and the search
        results from /v1/search are also pre-compressed.

        With 1 user message and 1 search iteration, exactly 1 compress API
        call should happen (for the initial user message). Any additional
        calls are wasted latency from re-compressing already-compressed text.
        """
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()

        original_create = MagicMock(
            side_effect=[
                _search_tool_use_response("latest news"),
                _text_response("Done."),
            ],
        )
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo
            mock_ttc.search.return_value = _make_search_response()

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "What is the news?"}],
            )

        # The whole conversation is compressed in a single call before the
        # provider request. The search loop must NOT compress again: the
        # initial messages were already compressed and /v1/search results
        # arrive pre-compressed.
        total_compress_calls = mock_ttc.compress_chat.call_count
        assert total_compress_calls == 1, (
            f"Expected 1 chat-compress API call (initial turn only), "
            f"got {total_compress_calls}. The search loop is re-compressing "
            "already-compressed messages, adding unnecessary latency."
        )

    def test_search_compression_stats_tracked(self) -> None:
        """Compression stats should include token savings from web search.

        The /v1/search endpoint compresses results server-side (1000 → 700
        tokens = 300 saved). These savings must be visible in
        client.compression.total_tokens_saved, separate from message
        compression savings.
        """
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()

        original_create = MagicMock(
            side_effect=[
                _search_tool_use_response("test query"),
                _text_response("Answer."),
            ],
        )
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            # Make message compression a no-op so we isolate search savings
            mock_ttc.compress_chat.side_effect = _chat_noop
            # Search returns 1000 input tokens compressed to 700 = 300 saved
            mock_ttc.search.return_value = SearchResponse(
                results=[
                    SearchResult(
                        url="https://example.com",
                        title="Big Article",
                        content="Compressed article content...",
                        score=0.95,
                    ),
                ],
                query="test query",
                original_input_tokens=1000,
                output_tokens=700,
            )

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Search something"}],
            )

        # Message compression saved 0 tokens (no-op mock). The search
        # endpoint saved 300 tokens. If total_tokens_saved is 0, the search
        # savings are not being tracked.
        assert wrapped.compression.total_tokens_saved >= 300, (
            f"total_tokens_saved is {wrapped.compression.total_tokens_saved}, "
            "expected at least 300 from search compression. The /v1/search "
            "endpoint's compression savings are not being tracked in "
            "CompressionStats."
        )


def _eager_search_model():
    """An Anthropic stub that wants to search on EVERY turn, but (like a real
    model) stops and answers once a search comes back as an error — which is how
    the budget cap signals 'no more searches' per query."""
    from unittest.mock import MagicMock

    def create(**kwargs):
        last = (kwargs.get("messages") or [])[-1:] or [None]
        content = last[0].get("content") if isinstance(last[0], dict) else None
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error")
            for b in content
        ):
            return _text_response("Final answer.")  # budget hit -> answer
        return _MockResponse(
            stop_reason="tool_use",
            content=[_ContentBlock(type="tool_use", name="ttc_web_search",
                                   id="toolu_x", input={"query": "q"})],
        )

    client = MagicMock()
    client.messages.create = create
    return client


class TestWebSearchMaxUses:
    def test_explicit_cap_stops_runaway_loop(self) -> None:
        from thetokencompany.anthropic import with_compression

        client = _eager_search_model()
        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo
            mock_ttc.search.return_value = _make_search_response()

            wrapped = with_compression(client, compression_api_key="ttc-test",
                                       web_search=True, web_search_max_uses=3)
            resp = wrapped.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": "research deeply"}],
            )

        # The model would search forever; the cap stops it at exactly 3.
        assert mock_ttc.search.call_count == 3
        assert resp.stop_reason == "end_turn"  # forced to answer

    def test_default_cap_applies_without_arg(self) -> None:
        from thetokencompany.anthropic import DEFAULT_WEB_SEARCH_MAX_USES, with_compression

        client = _eager_search_model()
        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo
            mock_ttc.search.return_value = _make_search_response()

            wrapped = with_compression(client, compression_api_key="ttc-test", web_search=True)
            wrapped.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": "research"}],
            )

        assert mock_ttc.search.call_count == DEFAULT_WEB_SEARCH_MAX_USES

    def test_native_max_uses_is_honored(self) -> None:
        from thetokencompany.anthropic import with_compression

        client = _eager_search_model()
        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo
            mock_ttc.search.return_value = _make_search_response()

            wrapped = with_compression(client, compression_api_key="ttc-test", web_search=True)
            # Caller passes the native web_search tool with its own max_uses=2;
            # web_search=True swaps in ttc_web_search but keeps the cap.
            wrapped.messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": "research"}],
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}],
            )

        assert mock_ttc.search.call_count == 2


# ---------------------------------------------------------------------------
# Graceful degradation: a compression-backend fault must never break the
# customer's underlying LLM call.
# ---------------------------------------------------------------------------


class TestCompressionGracefulDegradation:
    def test_sync_compress_failure_falls_through_uncompressed(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value=_text_response("ok"))
        mock_client.messages.create = original_create
        original_messages = [{"role": "user", "content": "hello"}]

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = RuntimeError("backend 503")

            wrapped = with_compression(mock_client, compression_api_key="ttc-test")
            result = wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=original_messages,
            )

        # The customer's call still succeeds, with the ORIGINAL messages.
        assert result.stop_reason == "end_turn"
        assert original_create.call_args[1]["messages"] == original_messages

    @pytest.mark.asyncio
    async def test_async_compress_failure_falls_through_uncompressed(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = AsyncMock(return_value=_text_response("ok"))
        mock_client.messages.create = original_create
        original_messages = [{"role": "user", "content": "hello"}]

        with patch("thetokencompany.anthropic.AsyncTheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat = AsyncMock(side_effect=RuntimeError("backend 503"))

            wrapped = with_compression(mock_client, compression_api_key="ttc-test")
            result = await wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=original_messages,
            )

        assert result.stop_reason == "end_turn"
        assert original_create.call_args[1]["messages"] == original_messages


# ---------------------------------------------------------------------------
# Integration tests for async wrapper
# ---------------------------------------------------------------------------


class TestWebSearchAsyncWrapper:
    @pytest.mark.asyncio
    async def test_web_search_false_no_change(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_text_response(),
        )

        with patch(
            "thetokencompany.anthropic.AsyncTheTokenCompany",
        ) as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat = AsyncMock(side_effect=_chat_echo)

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=False,
            )
            result = await wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"name": "calculator", "input_schema": {}}],
            )

        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_async_search_loop(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = AsyncMock(
            side_effect=[
                _search_tool_use_response("async query"),
                _text_response("Async result."),
            ],
        )
        mock_client.messages.create = original_create

        with patch(
            "thetokencompany.anthropic.AsyncTheTokenCompany",
        ) as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat = AsyncMock(side_effect=_chat_echo)
            mock_ttc.search = AsyncMock(
                return_value=_make_search_response(query="async query"),
            )

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=True,
            )
            result = await wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Search async"}],
            )

        # original_create was captured by the wrapper closure
        assert original_create.call_count == 2
        assert result.stop_reason == "end_turn"
        assert result.content[0].text == "Async result."
        mock_ttc.search.assert_awaited_once_with("async query")
