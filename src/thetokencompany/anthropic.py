"""Auto-compress messages for Anthropic clients."""

from __future__ import annotations

import functools
import inspect
from typing import Any

import httpx

from thetokencompany._async_client import AsyncTheTokenCompany
from thetokencompany._client import TheTokenCompany
from thetokencompany._compress import (
    DEFAULT_AGGRESSIVENESS,
    Aggressiveness,
    _AsyncStatsTTC,
    _compress_text_blocks,
    _compress_text_blocks_async,
    _resolve_aggressiveness,
    _StatsTTC,
    compress_anthropic_messages,
    compress_anthropic_messages_async,
    compress_text,
    compress_text_async,
)
from thetokencompany._constants import BASE_URL, BEAR_2
from thetokencompany._types import CompressionStats


def with_compression(
    client: Any,
    *,
    compression_api_key: str,
    model: str = BEAR_2,
    aggressiveness: Aggressiveness = DEFAULT_AGGRESSIVENESS,
    compress_assistant: bool = False,
    strip_server_tool_results: bool = False,
    base_url: str = BASE_URL,
    app_id: str | None = None,
    http_client: httpx.Client | None = None,
    async_http_client: httpx.AsyncClient | None = None,
) -> Any:
    """Wrap an Anthropic client to auto-compress messages.

    Compresses the ``system`` parameter and all non-assistant messages.

    Compression stats are available on ``client.compression``::

        client = with_compression(Anthropic(), compression_api_key="ttc-...")
        client.messages.create(...)
        print(client.compression.total_tokens_saved)

    Args:
        compress_assistant: When ``True``, also compress text blocks in
            assistant messages. Useful for multi-turn conversations where
            previous assistant responses (e.g. from web search) are large.
            Defaults to ``False`` to preserve the provider's KV cache.
        strip_server_tool_results: When ``True``, remove server-side tool
            result blocks (e.g. ``web_search_tool_result``) from assistant
            messages before sending. This can significantly reduce input
            tokens in multi-turn conversations that use server-side tools.
            Note: this disables citations in subsequent turns.
    """
    role_aggr = _resolve_aggressiveness(aggressiveness)
    if compress_assistant and "assistant" not in role_aggr:
        role_aggr["assistant"] = role_aggr.get("user", DEFAULT_AGGRESSIVENESS)
    system_aggr = role_aggr.get("system")
    stats = CompressionStats()
    original_create = client.messages.create

    if inspect.iscoroutinefunction(original_create):
        async_ttc = AsyncTheTokenCompany(
            api_key=compression_api_key,
            base_url=base_url,
            app_id=app_id,
            http_client=async_http_client,
        )
        compressor: Any = _AsyncStatsTTC(async_ttc, stats)

        @functools.wraps(original_create)
        async def async_create(*args: Any, **kwargs: Any) -> Any:
            stats._start_turn()
            if "messages" in kwargs:
                kwargs["messages"] = await compress_anthropic_messages_async(
                    compressor, kwargs["messages"], model, role_aggr,
                    strip_server_tool_results=strip_server_tool_results,
                )
            if system_aggr is not None and "system" in kwargs:
                system = kwargs["system"]
                if isinstance(system, str):
                    kwargs["system"] = await compress_text_async(
                        compressor, system, model, system_aggr
                    )
                elif isinstance(system, list):
                    kwargs["system"] = await _compress_text_blocks_async(
                        compressor, system, model, system_aggr
                    )
            stats._end_turn()
            return await original_create(*args, **kwargs)

        client.messages.create = async_create
    else:
        sync_ttc = TheTokenCompany(
            api_key=compression_api_key, base_url=base_url, app_id=app_id,
            http_client=http_client,
        )
        compressor = _StatsTTC(sync_ttc, stats)

        @functools.wraps(original_create)
        def sync_create(*args: Any, **kwargs: Any) -> Any:
            stats._start_turn()
            if "messages" in kwargs:
                kwargs["messages"] = compress_anthropic_messages(
                    compressor, kwargs["messages"], model, role_aggr,
                    strip_server_tool_results=strip_server_tool_results,
                )
            if system_aggr is not None and "system" in kwargs:
                system = kwargs["system"]
                if isinstance(system, str):
                    kwargs["system"] = compress_text(
                        compressor, system, model, system_aggr
                    )
                elif isinstance(system, list):
                    kwargs["system"] = _compress_text_blocks(
                        compressor, system, model, system_aggr
                    )
            stats._end_turn()
            return original_create(*args, **kwargs)

        client.messages.create = sync_create

    client.compression = stats
    return client
