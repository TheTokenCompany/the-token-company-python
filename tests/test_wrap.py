"""Tests for with_compression wrappers (OpenAI, Anthropic)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thetokencompany._compress import _resolve_aggressiveness
from thetokencompany._types import ChatCompressResponse, CompressResponse


def _mock_compress_response(text: str) -> CompressResponse:
    return CompressResponse(output=f"[compressed]{text}", output_tokens=5, input_tokens=20)


def _chat_echo(messages: list, **kwargs: object) -> ChatCompressResponse:
    """Mimic the server: echo the messages back (compression happens server-side
    now, so the wrapper's job is just to forward what the endpoint returns)."""
    return ChatCompressResponse(
        messages=messages,
        system=kwargs.get("system"),
        input_tokens=20,
        output_tokens=5,
        cache_hits=0,
        cache_misses=len(messages),
        compression_time=0.0,
    )


class TestResolveAggressiveness:
    def test_float_expands_to_default_roles(self) -> None:
        result = _resolve_aggressiveness(0.3)
        assert result == {"user": 0.3, "system": 0.3, "tool": 0.3, "assistant": 0.3}

    def test_dict_passed_through(self) -> None:
        d = {"user": 0.5, "system": 0.1}
        assert _resolve_aggressiveness(d) is d


# ---------------------------------------------------------------------------
# OpenAI message compression
# ---------------------------------------------------------------------------


class TestOpenAIMessages:
    def test_assistant_unchanged(self) -> None:
        from thetokencompany._compress import compress_openai_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "bye"},
            ]
            ttc = MagicMock()
            result = compress_openai_messages(
                ttc, messages, "bear-2", {"user": 0.2, "system": 0.2, "tool": 0.2}
            )

        assert result[0]["content"] == "[c]hello"
        assert result[1]["content"] == "hi there"
        assert result[2]["content"] == "[c]bye"

    def test_assistant_compressed_when_role_present(self) -> None:
        from thetokencompany._compress import compress_openai_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "a long agent response"},
            ]
            ttc = MagicMock()
            result = compress_openai_messages(
                ttc, messages, "bear-2", {"user": 0.2, "assistant": 0.3}
            )

        assert result[0]["content"] == "[c]hello"
        assert result[1]["content"] == "[c]a long agent response"

    def test_tool_role_compressed(self) -> None:
        from thetokencompany._compress import compress_openai_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [{"role": "tool", "content": "tool output"}]
            ttc = MagicMock()
            result = compress_openai_messages(ttc, messages, "bear-2", {"tool": 0.2})

        assert result[0]["content"] == "[c]tool output"

    def test_function_role_uses_tool_aggressiveness(self) -> None:
        from thetokencompany._compress import compress_openai_messages

        captured: list[float] = []

        def _track(_t: object, text: str, _m: str, aggr: float) -> str:
            captured.append(aggr)
            return text

        with patch("thetokencompany._compress.compress_text", side_effect=_track):
            messages = [{"role": "function", "content": "fn output"}]
            ttc = MagicMock()
            compress_openai_messages(ttc, messages, "bear-2", {"tool": 0.7})

        assert captured == [0.7]

    def test_system_role_compressed(self) -> None:
        from thetokencompany._compress import compress_openai_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [{"role": "system", "content": "you are helpful"}]
            ttc = MagicMock()
            result = compress_openai_messages(ttc, messages, "bear-2", {"system": 0.1})

        assert result[0]["content"] == "[c]you are helpful"

    def test_multimodal_blocks(self) -> None:
        from thetokencompany._compress import compress_openai_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "https://..."}},
                    ],
                }
            ]
            ttc = MagicMock()
            result = compress_openai_messages(ttc, messages, "bear-2", {"user": 0.2})

        blocks = result[0]["content"]
        assert blocks[0] == {"type": "text", "text": "[c]describe this"}
        assert blocks[1] == {"type": "image_url", "image_url": {"url": "https://..."}}

    def test_role_not_in_dict_skipped(self) -> None:
        from thetokencompany._compress import compress_openai_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": "hello"},
            ]
            ttc = MagicMock()
            result = compress_openai_messages(ttc, messages, "bear-2", {"user": 0.3})

        assert result[0]["content"] == "sys prompt"
        assert result[1]["content"] == "[c]hello"

    def test_per_role_aggressiveness(self) -> None:
        from thetokencompany._compress import compress_openai_messages

        captured: list[float] = []

        def _track(_t: object, text: str, _m: str, aggr: float) -> str:
            captured.append(aggr)
            return text

        with patch("thetokencompany._compress.compress_text", side_effect=_track):
            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "usr"},
            ]
            ttc = MagicMock()
            compress_openai_messages(ttc, messages, "bear-2", {"system": 0.1, "user": 0.7})

        assert captured == [0.1, 0.7]


# ---------------------------------------------------------------------------
# Anthropic message compression
# ---------------------------------------------------------------------------


class TestAnthropicMessages:
    def test_assistant_unchanged(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
            ttc = MagicMock()
            result = compress_anthropic_messages(
                ttc, messages, "bear-2", {"user": 0.2, "tool": 0.2}
            )

        assert result[0]["content"] == "[c]hello"
        assert result[1]["content"] == "hi there"

    def test_assistant_compressed_when_role_present(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "a long agent response"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "a text block"}],
                },
            ]
            ttc = MagicMock()
            result = compress_anthropic_messages(
                ttc, messages, "bear-2", {"user": 0.2, "assistant": 0.3}
            )

        assert result[0]["content"] == "[c]hello"
        assert result[1]["content"] == "[c]a long agent response"
        assert result[2]["content"][0]["text"] == "[c]a text block"

    def test_tool_result_blocks_compressed(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "search returned 500 results...",
                        }
                    ],
                }
            ]
            ttc = MagicMock()
            result = compress_anthropic_messages(ttc, messages, "bear-2", {"tool": 0.3})

        block = result[0]["content"][0]
        assert block["content"] == "[c]search returned 500 results..."
        assert block["tool_use_id"] == "toolu_123"

    def test_tool_result_with_content_blocks(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_456",
                            "content": [{"type": "text", "text": "detailed output"}],
                        }
                    ],
                }
            ]
            ttc = MagicMock()
            result = compress_anthropic_messages(ttc, messages, "bear-2", {"tool": 0.2})

        inner = result[0]["content"][0]["content"]
        assert inner == [{"type": "text", "text": "[c]detailed output"}]

    def test_mixed_text_and_tool_result(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        captured: list[tuple[str, float]] = []

        def _track(_t: object, text: str, _m: str, aggr: float) -> str:
            captured.append((text, aggr))
            return f"[c]{text}"

        with patch("thetokencompany._compress.compress_text", side_effect=_track):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "here are results"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "tool data",
                        },
                    ],
                }
            ]
            ttc = MagicMock()
            compress_anthropic_messages(ttc, messages, "bear-2", {"user": 0.1, "tool": 0.8})

        assert captured == [("here are results", 0.1), ("tool data", 0.8)]

    def test_tool_not_in_dict_skips_tool_result(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "raw tool data",
                        },
                    ],
                }
            ]
            ttc = MagicMock()
            result = compress_anthropic_messages(ttc, messages, "bear-2", {"user": 0.2})

        blocks = result[0]["content"]
        assert blocks[0]["text"] == "[c]hello"
        assert blocks[1]["content"] == "raw tool data"


# ---------------------------------------------------------------------------
# Wrapper integration tests
# ---------------------------------------------------------------------------


class TestOpenAIWrapper:
    def test_sync(self) -> None:
        from thetokencompany.openai import with_compression

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value="response")

        with patch("thetokencompany.openai.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo

            wrapped = with_compression(mock_client, compression_api_key="ttc-test")
            result = wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "world"},
                ],
            )

        assert result == "response"
        # The whole conversation goes in a single call, not one per message.
        assert mock_ttc.compress_chat.call_count == 1
        assert mock_ttc.compress_chat.call_args.kwargs["fmt"] == "openai"

    def test_assistant_compressed_by_default(self) -> None:
        from thetokencompany.openai import with_compression

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value="response")

        with patch("thetokencompany.openai.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo

            wrapped = with_compression(mock_client, compression_api_key="ttc-test")
            wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        # Default (scalar) aggressiveness must send the assistant role too.
        sent = mock_ttc.compress_chat.call_args.kwargs
        assert sent["aggressiveness"]["assistant"] == 0.2

    def test_dict_without_assistant_omits_it(self) -> None:
        from thetokencompany.openai import with_compression

        mock_client = MagicMock()
        mock_client.chat.completions.create = MagicMock(return_value="response")

        with patch("thetokencompany.openai.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                aggressiveness={"user": 0.2, "system": 0.2, "tool": 0.2},
            )
            wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

        # KV-cache opt-out: a per-role dict without "assistant" excludes it.
        sent = mock_ttc.compress_chat.call_args.kwargs
        assert "assistant" not in sent["aggressiveness"]

    @pytest.mark.asyncio
    async def test_async(self) -> None:
        from thetokencompany.openai import with_compression

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value="response")

        with patch("thetokencompany.openai.AsyncTheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat = AsyncMock(side_effect=_chat_echo)

            wrapped = with_compression(mock_client, compression_api_key="ttc-test")
            result = await wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result == "response"
        mock_ttc.compress_chat.assert_awaited_once()


class TestAnthropicWrapper:
    def test_compresses_system_string(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value="response")
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.return_value = ChatCompressResponse(
                messages=[{"role": "user", "content": "[c]hello"}],
                system="[compressed]sys prompt",
                input_tokens=40,
                output_tokens=10,
                cache_hits=0,
                cache_misses=2,
                compression_time=0.0,
            )

            wrapped = with_compression(mock_client, compression_api_key="ttc-test")
            result = wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system="You are a helpful assistant with lots of context...",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result == "response"
        # System + messages compressed together in ONE call.
        mock_ttc.compress_chat.assert_called_once()
        sent = mock_ttc.compress_chat.call_args.kwargs
        assert sent["system"] == "You are a helpful assistant with lots of context..."
        assert sent["fmt"] == "anthropic"
        # The compressed system from the response is forwarded to the provider.
        assert original_create.call_args[1]["system"] == "[compressed]sys prompt"

    def test_assistant_messages_preserved(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value="response")
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo

            wrapped = with_compression(mock_client, compression_api_key="ttc-test")
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "original response"},
                    {"role": "user", "content": "follow up"},
                ],
            )

        # Whole conversation forwarded in one call; assistant left intact
        # (the server preserves it — by default it's absent from role_aggr).
        assert mock_ttc.compress_chat.call_count == 1
        call_kwargs = original_create.call_args[1]
        assert call_kwargs["messages"][1]["content"] == "original response"


class TestAnthropicServerToolStripping:
    def test_strip_server_tool_results_removes_both_block_types(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_123",
                        "name": "web_search",
                        "input": {"query": "test"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_123",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com",
                                "title": "Example",
                                "encrypted_content": "abc123",
                            }
                        ],
                    },
                    {"type": "text", "text": "Here are the results."},
                ],
            },
        ]
        ttc = MagicMock()
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"user": 0.2, "tool": 0.2},
            strip_server_tool_results=True,
        )

        blocks = result[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Here are the results."

    def test_strip_disabled_preserves_all_blocks(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_123",
                        "name": "web_search",
                        "input": {"query": "test"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_123",
                        "content": [],
                    },
                    {"type": "text", "text": "Results."},
                ],
            },
        ]
        ttc = MagicMock()
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"user": 0.2},
            strip_server_tool_results=False,
        )

        blocks = result[0]["content"]
        assert len(blocks) == 3

    def test_strip_all_blocks_fallback_preserves_original(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_123",
                        "name": "web_search",
                        "input": {"query": "test"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_123",
                        "content": [],
                    },
                ],
            },
        ]
        ttc = MagicMock()
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"user": 0.2},
            strip_server_tool_results=True,
        )

        blocks = result[0]["content"]
        assert len(blocks) == 2

    def test_user_messages_unaffected_by_strip(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
                        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
                        {"type": "text", "text": "answer"},
                    ],
                },
                {"role": "user", "content": "follow up"},
            ]
            ttc = MagicMock()
            result = compress_anthropic_messages(
                ttc,
                messages,
                "bear-2",
                {"user": 0.2, "tool": 0.2},
                strip_server_tool_results=True,
            )

        assert result[0]["content"] == "[c]hello"
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "text"
        assert result[2]["content"] == "[c]follow up"


class TestAnthropicAssistantCompression:
    def test_compress_assistant_text_blocks(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "A long assistant response with lots of content."},
                    ],
                },
            ]
            ttc = MagicMock()
            result = compress_anthropic_messages(
                ttc,
                messages,
                "bear-2",
                {"user": 0.2, "assistant": 0.3},
            )

        assert (
            result[0]["content"][0]["text"] == "[c]A long assistant response with lots of content."
        )

    def test_assistant_string_content_compressed(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        with patch("thetokencompany._compress.compress_text") as mock_ct:
            mock_ct.side_effect = lambda _t, text, _m, _a: f"[c]{text}"
            messages = [
                {"role": "assistant", "content": "Plain text assistant response."},
            ]
            ttc = MagicMock()
            result = compress_anthropic_messages(
                ttc,
                messages,
                "bear-2",
                {"assistant": 0.2},
            )

        assert result[0]["content"] == "[c]Plain text assistant response."

    def test_no_assistant_aggr_skips_compression(self) -> None:
        from thetokencompany._compress import compress_anthropic_messages

        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "should not change"}],
            },
        ]
        ttc = MagicMock()
        result = compress_anthropic_messages(
            ttc,
            messages,
            "bear-2",
            {"user": 0.2},
        )

        assert result[0]["content"][0]["text"] == "should not change"


class TestAnthropicWrapperOptions:
    def test_compress_assistant_sets_aggressiveness(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value="response")
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                compress_assistant=True,
            )
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "long assistant text to compress"},
                    {"role": "user", "content": "follow up"},
                ],
            )

        # compress_assistant=True must add "assistant" to the per-role map sent
        # to the server, so the server compresses assistant text too.
        sent = mock_ttc.compress_chat.call_args.kwargs
        assert "assistant" in sent["aggressiveness"]

    def test_assistant_compressed_by_default(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        mock_client.messages.create = MagicMock(return_value="response")

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo

            wrapped = with_compression(mock_client, compression_api_key="ttc-test")
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "long agent text"},
                    {"role": "user", "content": "follow up"},
                ],
            )

        # No compress_assistant flag needed — assistant is on by default now.
        sent = mock_ttc.compress_chat.call_args.kwargs
        assert sent["aggressiveness"]["assistant"] == 0.2

    def test_strip_server_tool_results_in_wrapper(self) -> None:
        from thetokencompany.anthropic import with_compression

        mock_client = MagicMock()
        original_create = MagicMock(return_value="response")
        mock_client.messages.create = original_create

        with patch("thetokencompany.anthropic.TheTokenCompany") as MockTTC:
            mock_ttc = MockTTC.return_value
            mock_ttc.compress_chat.side_effect = _chat_echo

            wrapped = with_compression(
                mock_client,
                compression_api_key="ttc-test",
                strip_server_tool_results=True,
            )
            wrapped.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "s1",
                                "name": "web_search",
                                "input": {},
                            },
                            {
                                "type": "web_search_tool_result",
                                "tool_use_id": "s1",
                                "content": [
                                    {
                                        "type": "web_search_result",
                                        "encrypted_content": "long_encrypted_data_" * 100,
                                    }
                                ],
                            },
                            {"type": "text", "text": "Here is the answer."},
                        ],
                    },
                    {"role": "user", "content": "follow up"},
                ],
            )

        # The strip flag must be forwarded to the server, which performs the
        # actual block removal (see the API's chat_compress service tests).
        sent = mock_ttc.compress_chat.call_args.kwargs
        assert sent["strip_server_tool_results"] is True


class TestHelpers:
    def test_empty_content_passthrough(self) -> None:
        from thetokencompany._compress import compress_text

        ttc = MagicMock()
        assert compress_text(ttc, "", "bear-2", 0.2) == ""
        assert compress_text(ttc, "   ", "bear-2", 0.2) == "   "
        ttc.compress.assert_not_called()
