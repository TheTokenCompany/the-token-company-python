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
from thetokencompany._types import CompressionStats, SearchResponse

# ---------------------------------------------------------------------------
# Web-search tool definition & helpers
# ---------------------------------------------------------------------------

_TTC_SEARCH_TOOL: dict[str, Any] = {
    "name": "ttc_web_search",
    "description": (
        "Search the web for current information. Use this when you need "
        "up-to-date facts, prices, news, or any information that may have "
        "changed after your training cutoff."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            }
        },
        "required": ["query"],
    },
}


def _inject_search_tool(kwargs: dict[str, Any]) -> None:
    """Remove Anthropic's server-side web search and inject our tool."""
    tools = list(kwargs.get("tools", []))
    # Remove Anthropic's server-side web search if present
    tools = [t for t in tools if not str(t.get("type", "")).startswith("web_search_")]
    # Add our tool if not already there
    if not any(t.get("name") == "ttc_web_search" for t in tools):
        tools.append(_TTC_SEARCH_TOOL)
    kwargs["tools"] = tools


def _has_ttc_search_use(response: Any) -> bool:
    """Return True if the response contains a ttc_web_search tool_use."""
    if response.stop_reason != "tool_use":
        return False
    return any(b.type == "tool_use" and b.name == "ttc_web_search" for b in response.content)


def _format_search_results(search_response: SearchResponse) -> str:
    """Format search results as plain text for the tool_result block."""
    lines: list[str] = []
    for r in search_response.results:
        lines.append(f"Source: {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append(r.content)
        lines.append("")
    return "\n".join(lines)


def _handle_search_loop_sync(
    response: Any,
    kwargs: dict[str, Any],
    original_create: Any,
    ttc_client: TheTokenCompany,
    model: str,
    stats: CompressionStats,
    compressor: Any,
    role_aggr: dict[str, float],
    system_aggr: float | None,
    strip_server_tool_results: bool,
) -> Any:
    """Handle the ttc_web_search tool-use loop synchronously."""
    messages = list(kwargs.get("messages", []))
    while _has_ttc_search_use(response):
        assistant_content = [b.model_dump() for b in response.content]
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "ttc_web_search":
                query = block.input.get("query", "")
                search_result = ttc_client.search(query)
                stats._record_search(search_result)
                result_text = _format_search_results(search_result)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        new_kwargs = {**kwargs, "messages": messages}
        response = original_create(**new_kwargs)

    return response


async def _handle_search_loop_async(
    response: Any,
    kwargs: dict[str, Any],
    original_create: Any,
    ttc_client: AsyncTheTokenCompany,
    model: str,
    stats: CompressionStats,
    compressor: Any,
    role_aggr: dict[str, float],
    system_aggr: float | None,
    strip_server_tool_results: bool,
) -> Any:
    """Handle the ttc_web_search tool-use loop asynchronously."""
    messages = list(kwargs.get("messages", []))
    while _has_ttc_search_use(response):
        assistant_content = [b.model_dump() for b in response.content]
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "ttc_web_search":
                query = block.input.get("query", "")
                search_result = await ttc_client.search(query)
                stats._record_search(search_result)
                result_text = _format_search_results(search_result)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        new_kwargs = {**kwargs, "messages": messages}
        response = await original_create(**new_kwargs)

    return response


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------


def with_compression(
    client: Any,
    *,
    compression_api_key: str,
    model: str = BEAR_2,
    aggressiveness: Aggressiveness = DEFAULT_AGGRESSIVENESS,
    compress_assistant: bool = False,
    strip_server_tool_results: bool = False,
    web_search: bool = False,
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
        web_search: When ``True``, intercept Anthropic's server-side
            ``web_search_*`` server-side tools and replace them with a client-side
            tool backed by TTC's ``/v1/search`` endpoint.  Search results
            are automatically compressed before being fed back to the model.
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
            if web_search:
                _inject_search_tool(kwargs)
            if "messages" in kwargs:
                kwargs["messages"] = await compress_anthropic_messages_async(
                    compressor,
                    kwargs["messages"],
                    model,
                    role_aggr,
                    strip_server_tool_results=strip_server_tool_results,
                    skip_tool_name="ttc_web_search" if web_search else None,
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
            response = await original_create(*args, **kwargs)

            if web_search:
                response = await _handle_search_loop_async(
                    response,
                    kwargs,
                    original_create,
                    async_ttc,
                    model,
                    stats,
                    compressor,
                    role_aggr,
                    system_aggr,
                    strip_server_tool_results,
                )

            return response

        client.messages.create = async_create
    else:
        sync_ttc = TheTokenCompany(
            api_key=compression_api_key,
            base_url=base_url,
            app_id=app_id,
            http_client=http_client,
        )
        compressor = _StatsTTC(sync_ttc, stats)

        @functools.wraps(original_create)
        def sync_create(*args: Any, **kwargs: Any) -> Any:
            stats._start_turn()
            if web_search:
                _inject_search_tool(kwargs)
            if "messages" in kwargs:
                kwargs["messages"] = compress_anthropic_messages(
                    compressor,
                    kwargs["messages"],
                    model,
                    role_aggr,
                    strip_server_tool_results=strip_server_tool_results,
                    skip_tool_name="ttc_web_search" if web_search else None,
                )
            if system_aggr is not None and "system" in kwargs:
                system = kwargs["system"]
                if isinstance(system, str):
                    kwargs["system"] = compress_text(compressor, system, model, system_aggr)
                elif isinstance(system, list):
                    kwargs["system"] = _compress_text_blocks(compressor, system, model, system_aggr)
            stats._end_turn()
            response = original_create(*args, **kwargs)

            if web_search:
                response = _handle_search_loop_sync(
                    response,
                    kwargs,
                    original_create,
                    sync_ttc,
                    model,
                    stats,
                    compressor,
                    role_aggr,
                    system_aggr,
                    strip_server_tool_results,
                )

            return response

        client.messages.create = sync_create

    client.compression = stats
    return client
