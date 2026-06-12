"""Tests for skipping compression of pre-compressed search tool results."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from thetokencompany._compress import compress_anthropic_messages
from thetokencompany._types import CompressResponse


def _mock_ttc() -> MagicMock:
    ttc = MagicMock()
    ttc.compress.side_effect = lambda text, **kw: CompressResponse(
        output=f"[c]{text}",
        output_tokens=5,
        input_tokens=20,
    )
    return ttc


class TestSkipToolName:
    def test_skips_search_tool_result(self) -> None:
        ttc = _mock_ttc()
        messages = [
            {"role": "user", "content": "search for news"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {
                        "type": "tool_use",
                        "id": "toolu_s1",
                        "name": "ttc_web_search",
                        "input": {"query": "news"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_s1",
                        "content": "Already compressed search content",
                    },
                ],
            },
        ]
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"user": 0.2, "tool": 0.3},
            skip_tool_name="ttc_web_search",
        )
        assert result[0]["content"] == "[c]search for news"
        assert result[2]["content"][0]["content"] == "Already compressed search content"

    def test_still_compresses_non_search_tool_results(self) -> None:
        ttc = _mock_ttc()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_s1",
                        "name": "ttc_web_search",
                        "input": {"query": "q"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_c1",
                        "name": "calculator",
                        "input": {"expr": "2+2"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_s1", "content": "search data"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_c1",
                        "content": "calculator data",
                    },
                ],
            },
        ]
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"tool": 0.3},
            skip_tool_name="ttc_web_search",
        )
        blocks = result[1]["content"]
        assert blocks[0]["content"] == "search data"
        assert blocks[1]["content"] == "[c]calculator data"

    def test_skips_across_multiple_turns(self) -> None:
        ttc = _mock_ttc()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_s1",
                        "name": "ttc_web_search",
                        "input": {"query": "q1"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_s1", "content": "result 1"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Let me search more."}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_s2",
                        "name": "ttc_web_search",
                        "input": {"query": "q2"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_s2", "content": "result 2"},
                ],
            },
        ]
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"tool": 0.3},
            skip_tool_name="ttc_web_search",
        )
        assert result[1]["content"][0]["content"] == "result 1"
        assert result[4]["content"][0]["content"] == "result 2"

    def test_compresses_normally_without_skip_tool_name(self) -> None:
        ttc = _mock_ttc()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_s1",
                        "name": "ttc_web_search",
                        "input": {"query": "q"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_s1", "content": "search data"},
                ],
            },
        ]
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"tool": 0.3},
        )
        assert result[1]["content"][0]["content"] == "[c]search data"

    def test_works_after_json_serialization(self) -> None:
        """Simulates server switch / client recreation."""
        ttc = _mock_ttc()
        original = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "ttc_web_search",
                        "input": {"query": "test"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "pre-compressed result",
                    },
                ],
            },
            {"role": "user", "content": "follow up question"},
        ]
        messages = json.loads(json.dumps(original))
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"user": 0.2, "tool": 0.3},
            skip_tool_name="ttc_web_search",
        )
        assert result[1]["content"][0]["content"] == "pre-compressed result"
        assert result[2]["content"] == "[c]follow up question"

    def test_handles_tool_result_with_array_content(self) -> None:
        ttc = _mock_ttc()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_s1",
                        "name": "ttc_web_search",
                        "input": {"query": "q"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_s1",
                        "content": [{"type": "text", "text": "search result text"}],
                    },
                ],
            },
        ]
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"tool": 0.3},
            skip_tool_name="ttc_web_search",
        )
        inner = result[1]["content"][0]["content"]
        assert inner[0]["text"] == "search result text"

    def test_compresses_orphaned_tool_result(self) -> None:
        ttc = _mock_ttc()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_orphan", "content": "some data"},
                ],
            },
        ]
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"tool": 0.3},
            skip_tool_name="ttc_web_search",
        )
        assert result[0]["content"][0]["content"] == "[c]some data"


class TestSkipInSyncWrapper:
    def test_no_double_compress_in_multi_turn(self) -> None:
        from tests.test_web_search import (
            _make_search_response,
            _mock_compress_response,
            _search_tool_use_response,
            _text_response,
        )
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(
            side_effect=[
                _search_tool_use_response("news", "toolu_s1"),
                _text_response("Here is the news."),
            ],
        )
        mock_client.messages.create = original_create

        compress_calls: list[str] = []

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value

            def tracking_compress(text: str, **kw: Any) -> CompressResponse:
                compress_calls.append(text)
                return _mock_compress_response(text)

            mock_ttc.compress.side_effect = tracking_compress
            mock_ttc.search.return_value = _make_search_response(query="news")

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

        # The search result content should NOT have been compressed
        search_content_compressed = any(
            "Example content about the query" in c for c in compress_calls
        )
        assert not search_content_compressed

    def test_web_search_false_still_compresses_tool_results(self) -> None:
        from tests.test_web_search import _mock_compress_response, _text_response
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value=_text_response())
        mock_client.messages.create = original_create

        compress_calls: list[str] = []

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value

            def tracking_compress(text: str, **kw: Any) -> CompressResponse:
                compress_calls.append(text)
                return _mock_compress_response(text)

            mock_ttc.compress.side_effect = tracking_compress

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                web_search=False,
            )
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_s1",
                                "name": "ttc_web_search",
                                "input": {"query": "q"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_s1",
                                "content": "search data",
                            },
                        ],
                    },
                ],
            )

        assert any("search data" in c for c in compress_calls)
