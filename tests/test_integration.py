"""Integration tests hitting real TTC and Anthropic endpoints."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

TTC_API_KEY = os.environ.get("TTC_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
skip = not TTC_API_KEY or not ANTHROPIC_API_KEY


@pytest.mark.skipif(skip, reason="TTC_API_KEY and ANTHROPIC_API_KEY required")
class TestIntegration:
    def test_ttc_compress(self) -> None:
        from thetokencompany import TheTokenCompany

        ttc = TheTokenCompany(api_key=TTC_API_KEY)
        result = ttc.compress(
            "The quick brown fox jumps over the lazy dog."
            " This is a test of the compression system.",
            model="bear-2",
            aggressiveness=0.3,
        )
        assert result.output
        assert result.output_tokens <= result.input_tokens
        assert result.tokens_saved >= 0
        ttc.close()

    def test_ttc_search(self) -> None:
        from thetokencompany import TheTokenCompany

        ttc = TheTokenCompany(api_key=TTC_API_KEY)
        result = ttc.search("What is TypeScript?", max_results=3)
        assert len(result.results) > 0
        assert result.results[0].url
        assert result.results[0].title
        assert result.results[0].content
        assert result.tokens_saved >= 0
        ttc.close()

    def test_with_compression_wrapper(self) -> None:
        import anthropic

        from thetokencompany.anthropic import with_compression

        client = with_compression(
            anthropic.Anthropic(),
            compression_api_key=TTC_API_KEY,
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        assert response.stop_reason == "end_turn"
        assert response.content[0].text
        assert client.compression.total_tokens_saved >= 0

    def test_web_search_triggers_and_returns(self) -> None:
        import anthropic

        from thetokencompany.anthropic import with_compression

        client = with_compression(
            anthropic.Anthropic(),
            compression_api_key=TTC_API_KEY,
            web_search=True,
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system="You have a web search tool. Always use it to answer questions. Be very brief.",
            messages=[
                {
                    "role": "user",
                    "content": "What is the current population of Finland? Search for it.",
                }
            ],
        )
        assert response.stop_reason == "end_turn"
        text_block = next((b for b in response.content if b.type == "text"), None)
        assert text_block is not None

    def test_search_results_not_double_compressed_multi_turn(self) -> None:
        import anthropic

        from thetokencompany._client import TheTokenCompany as RealTTC
        from thetokencompany.anthropic import with_compression

        compress_inputs: list[str] = []
        real_compress = RealTTC.compress

        def tracking_compress(self_ttc: Any, text: str, **kwargs: Any) -> Any:
            compress_inputs.append(text)
            return real_compress(self_ttc, text, **kwargs)

        with patch.object(RealTTC, "compress", tracking_compress):
            client = with_compression(
                anthropic.Anthropic(),
                compression_api_key=TTC_API_KEY,
                web_search=True,
            )

            search_tool_use_id = "toolu_turn1"
            search_result_content = (
                "Source: Wikipedia\n"
                "URL: https://en.wikipedia.org/wiki/Reykjavik\n"
                "Reykjavik is the capital of Iceland."
            )
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": "What is the capital of Iceland?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": search_tool_use_id,
                            "name": "ttc_web_search",
                            "input": {"query": "capital of Iceland"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": search_tool_use_id,
                            "content": search_result_content,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The capital of Iceland is Reykjavik."}],
                },
                {"role": "user", "content": "And what is the population of that city?"},
            ]

            compress_inputs.clear()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system="You have a web search tool. You can use it if needed. Be very brief.",
                messages=messages,
            )

        assert response.stop_reason == "end_turn"
        search_was_compressed = any(
            "Reykjavik is the capital of Iceland" in t for t in compress_inputs
        )
        assert not search_was_compressed
        assert len(compress_inputs) > 0

    def test_regular_tool_results_still_compressed(self) -> None:
        import anthropic

        from thetokencompany._client import TheTokenCompany as RealTTC
        from thetokencompany.anthropic import with_compression

        compress_inputs: list[str] = []
        real_compress = RealTTC.compress

        def tracking_compress(self_ttc: Any, text: str, **kwargs: Any) -> Any:
            compress_inputs.append(text)
            return real_compress(self_ttc, text, **kwargs)

        with patch.object(RealTTC, "compress", tracking_compress):
            client = with_compression(
                anthropic.Anthropic(),
                compression_api_key=TTC_API_KEY,
                web_search=True,
            )

            messages: list[dict[str, Any]] = [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_calc",
                            "name": "calculator",
                            "input": {"expr": "2+2"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_calc",
                            "content": "The calculator returned the result: 4",
                        },
                    ],
                },
                {"role": "user", "content": "Thanks, what about 3+3?"},
            ]

            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=100,
                messages=messages,
            )

        # The calculator tool_result should have been sent to compress
        # (regardless of what the model responds with)
        calc_was_compressed = any("calculator returned the result" in t for t in compress_inputs)
        assert calc_was_compressed
