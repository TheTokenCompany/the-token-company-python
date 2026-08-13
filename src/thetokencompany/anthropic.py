"""Auto-compress messages for Anthropic clients."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any

import httpx

from thetokencompany._async_client import AsyncTheTokenCompany
from thetokencompany._client import TheTokenCompany
from thetokencompany._compress import (
    DEFAULT_AGGRESSIVENESS,
    Aggressiveness,
    _collect_tool_use_ids,
    _resolve_aggressiveness,
)
from thetokencompany._constants import BASE_URL, BEAR_2
from thetokencompany._types import CompressionStats, SearchResponse

logger = logging.getLogger("thetokencompany")

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


# Default cap on ttc_web_search calls per `messages.create`, mirroring
# Anthropic's native `web_search` `max_uses`. Without a cap the agent can loop
# indefinitely (each search is a sequential round-trip), which is the #1 source
# of runaway latency. 10 suits a research/search agent (Anthropic's threshold
# for "comparative or multi-entity research") while still hard-bounding latency.
# Drop to ~3 for latency-sensitive lookups; raise to ~20 for deep research.
# Override per call via `with_compression(web_search_max_uses=N)` or by setting
# `max_uses` on the native web_search tool you pass.
DEFAULT_WEB_SEARCH_MAX_USES = 10


def _effective_max_uses(explicit: int | None, native: int | None) -> int:
    """Resolve the search cap: explicit arg > native tool's max_uses > default."""
    if explicit is not None:
        return explicit
    if native is not None:
        return native
    return DEFAULT_WEB_SEARCH_MAX_USES


def _inject_search_tool(kwargs: dict[str, Any]) -> int | None:
    """Remove Anthropic's server-side web search and inject ttc_web_search.

    Returns the ``max_uses`` declared on any stripped native ``web_search_*``
    tool (so a caller's existing cap carries over unchanged), or None.
    """
    tools = list(kwargs.get("tools", []))
    native_max_uses: int | None = None
    kept: list[Any] = []
    for t in tools:
        if str(t.get("type", "")).startswith("web_search_"):
            if native_max_uses is None and isinstance(t.get("max_uses"), int):
                native_max_uses = t["max_uses"]
            continue  # strip the native tool
        kept.append(t)
    if not any(t.get("name") == "ttc_web_search" for t in kept):
        kept.append(_TTC_SEARCH_TOOL)
    kwargs["tools"] = kept
    return native_max_uses


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


def _search_ok(tool_use_id: str, search_response: SearchResponse) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": _format_search_results(search_response),
    }


def _search_failed(tool_use_id: str, err: Exception) -> dict[str, Any]:
    """Error result for a failed search (native returns an error result too, so
    the agent keeps going instead of crashing mid-run)."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "is_error": True,
        "content": f"Web search failed: {type(err).__name__}: {err}",
    }


def _run_searches_threaded(ttc_client: TheTokenCompany, blocks: list[Any]) -> list[Any]:
    """Run the round's searches CONCURRENTLY across threads (httpx.Client is
    thread-safe). Returns a SearchResponse or an Exception per block, in order."""
    def one(block: Any) -> Any:
        try:
            return ttc_client.search((block.input or {}).get("query", ""))
        except Exception as e:  # noqa: BLE001 — surfaced as an error tool_result
            return e

    if not blocks:
        return []
    if len(blocks) == 1:
        return [one(blocks[0])]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(len(blocks), 8)) as pool:
        return list(pool.map(one, blocks))


def _budget_split(
    search_blocks: list[Any], searches_done: int, max_uses: int | None
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    """Split a round's search blocks into the ones to actually run (within the
    remaining budget, in order) and pre-built budget-exhausted results for the
    rest."""
    to_run: list[Any] = []
    exhausted: dict[str, dict[str, Any]] = {}
    for block in search_blocks:
        if max_uses is not None and searches_done + len(to_run) >= max_uses:
            exhausted[block.id] = _search_budget_exhausted(block.id)
        else:
            to_run.append(block)
    return to_run, exhausted


def _search_budget_exhausted(tool_use_id: str) -> dict[str, Any]:
    """An error tool_result for a single over-budget search — like the
    'max uses exceeded' / failed-search result a native web-search tool returns.
    Scoped to THIS query (not a global change): the tool stays available, the
    model just sees that this search didn't run and answers with what it has."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "is_error": True,
        "content": (
            "This search was not performed: the web-search limit for this "
            "request has been reached. Do not retry; answer using the results "
            "already gathered."
        ),
    }


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
    max_uses: int | None,
) -> Any:
    """Handle the ttc_web_search tool-use loop synchronously.

    Caps the number of *actual* searches at ``max_uses`` (None = uncapped),
    mirroring Anthropic's native web search: once the budget is spent, any
    further search request gets an error tool_result (`is_error`, scoped to that
    one query) instead of running — so the expensive `/v1/search` calls are
    hard-bounded and the model answers with what it has. The tool stays
    available; nothing global changes."""
    messages = list(kwargs.get("messages", []))
    searches_done = 0
    while _has_ttc_search_use(response):
        messages.append({"role": "assistant",
                         "content": [b.model_dump() for b in response.content]})

        search_blocks = [b for b in response.content
                         if b.type == "tool_use" and b.name == "ttc_web_search"]
        to_run, results_by_id = _budget_split(search_blocks, searches_done, max_uses)

        # Run this round's searches concurrently (native does the same).
        outcomes = _run_searches_threaded(ttc_client, to_run)
        for block, outcome in zip(to_run, outcomes, strict=False):
            if isinstance(outcome, Exception):
                results_by_id[block.id] = _search_failed(block.id, outcome)
            else:
                stats._record_search(outcome)
                results_by_id[block.id] = _search_ok(block.id, outcome)
        searches_done += len(to_run)

        tool_results = [results_by_id[b.id] for b in search_blocks]
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        response = original_create(**{**kwargs, "messages": messages})

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
    max_uses: int | None,
) -> Any:
    """Handle the ttc_web_search tool-use loop asynchronously.

    Caps the number of actual searches at ``max_uses`` (None = uncapped); see
    the sync variant for the per-query, native-style semantics."""
    messages = list(kwargs.get("messages", []))
    searches_done = 0
    while _has_ttc_search_use(response):
        messages.append({"role": "assistant",
                         "content": [b.model_dump() for b in response.content]})

        search_blocks = [b for b in response.content
                         if b.type == "tool_use" and b.name == "ttc_web_search"]
        to_run, results_by_id = _budget_split(search_blocks, searches_done, max_uses)

        # Run this round's searches concurrently (native does the same).
        if to_run:
            outcomes = await asyncio.gather(
                *[ttc_client.search((b.input or {}).get("query", "")) for b in to_run],
                return_exceptions=True,
            )
            for block, outcome in zip(to_run, outcomes, strict=False):
                if isinstance(outcome, BaseException):
                    err = outcome if isinstance(outcome, Exception) else Exception(str(outcome))
                    results_by_id[block.id] = _search_failed(block.id, err)
                else:
                    stats._record_search(outcome)
                    results_by_id[block.id] = _search_ok(block.id, outcome)
            searches_done += len(to_run)

        tool_results = [results_by_id[b.id] for b in search_blocks]
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        response = await original_create(**{**kwargs, "messages": messages})

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
    web_search_max_uses: int | None = None,
    base_url: str = BASE_URL,
    app_id: str | None = None,
    http_client: httpx.Client | None = None,
    async_http_client: httpx.AsyncClient | None = None,
) -> Any:
    """Wrap an Anthropic client to auto-compress messages.

    Compresses the ``system`` parameter and all conversation turns — user,
    tool, and assistant/agent — by default. To keep the provider's KV cache
    warm, pass a per-role ``aggressiveness`` dict that omits the ``assistant``
    key (e.g. ``{"user": 0.2, "system": 0.2, "tool": 0.2}``).

    Compression stats are available on ``client.compression``::

        client = with_compression(Anthropic(), compression_api_key="ttc-...")
        client.messages.create(...)
        print(client.compression.total_tokens_saved)

    Args:
        compress_assistant: Deprecated / redundant — assistant (agent) turns
            are now compressed by default. Kept for back-compat: forces the
            ``assistant`` role back on when a per-role ``aggressiveness`` dict
            omits it. To exclude assistant turns, pass a dict without the
            ``assistant`` key instead.
        strip_server_tool_results: When ``True``, remove server-side tool
            result blocks (e.g. ``web_search_tool_result``) from assistant
            messages before sending. This can significantly reduce input
            tokens in multi-turn conversations that use server-side tools.
            Note: this disables citations in subsequent turns.
        web_search: When ``True``, intercept Anthropic's server-side
            ``web_search_*`` server-side tools and replace them with a client-side
            tool backed by TTC's ``/v1/search`` endpoint.  Search results
            are automatically compressed before being fed back to the model.
        web_search_max_uses: Cap on the number of ``ttc_web_search`` calls per
            ``messages.create``, mirroring Anthropic native web search's
            ``max_uses``. When the budget is spent, a further search returns an
            error result (``is_error``) scoped to that one query — exactly like
            native's ``max_uses_exceeded`` — so the model answers with what it
            has and the expensive ``/v1/search`` calls are bounded. The tool
            stays available; nothing global changes.

            Precedence: this argument, else ``max_uses`` on a native
            ``web_search`` tool you pass (so existing configs carry over), else
            ``DEFAULT_WEB_SEARCH_MAX_USES`` (10). Per Anthropic's guidance: simple
            lookups use 1–3 searches, so ``3`` is good for latency-sensitive
            apps; research agents should set ``15``–``20``. Pass ``0`` to disable
            search entirely.
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
        @functools.wraps(original_create)
        async def async_create(*args: Any, **kwargs: Any) -> Any:
            native_max_uses: int | None = None
            if web_search:
                native_max_uses = _inject_search_tool(kwargs)
            if "messages" in kwargs:
                # One request for messages + system; the server walks the roles
                # and tool_result blocks and serves re-sent history from cache.
                skip_ids = (
                    list(_collect_tool_use_ids(kwargs["messages"], "ttc_web_search"))
                    if web_search else None
                )
                try:
                    result = await async_ttc.compress_chat(
                        kwargs["messages"],
                        model=model,
                        fmt="anthropic",
                        aggressiveness=role_aggr,
                        system=kwargs.get("system"),
                        strip_server_tool_results=strip_server_tool_results,
                        skip_tool_use_ids=skip_ids,
                    )
                    kwargs["messages"] = result.messages
                    if "system" in kwargs:
                        kwargs["system"] = result.system
                    stats._record_chat(result)
                except Exception as exc:  # pragma: no cover - network/backend faults
                    # Graceful degradation: a compression-backend fault must never
                    # break the customer's underlying LLM call. Fall through with
                    # the original, uncompressed messages.
                    logger.warning(
                        "TTC compression failed (%s); sending uncompressed.", exc
                    )
            response = await original_create(*args, **kwargs)

            if web_search:
                response = await _handle_search_loop_async(
                    response,
                    kwargs,
                    original_create,
                    async_ttc,
                    model,
                    stats,
                    None,
                    role_aggr,
                    system_aggr,
                    strip_server_tool_results,
                    _effective_max_uses(web_search_max_uses, native_max_uses),
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
        @functools.wraps(original_create)
        def sync_create(*args: Any, **kwargs: Any) -> Any:
            native_max_uses: int | None = None
            if web_search:
                native_max_uses = _inject_search_tool(kwargs)
            if "messages" in kwargs:
                skip_ids = (
                    list(_collect_tool_use_ids(kwargs["messages"], "ttc_web_search"))
                    if web_search else None
                )
                try:
                    result = sync_ttc.compress_chat(
                        kwargs["messages"],
                        model=model,
                        fmt="anthropic",
                        aggressiveness=role_aggr,
                        system=kwargs.get("system"),
                        strip_server_tool_results=strip_server_tool_results,
                        skip_tool_use_ids=skip_ids,
                    )
                    kwargs["messages"] = result.messages
                    if "system" in kwargs:
                        kwargs["system"] = result.system
                    stats._record_chat(result)
                except Exception as exc:  # pragma: no cover - network/backend faults
                    # Graceful degradation: a compression-backend fault must never
                    # break the customer's underlying LLM call. Fall through with
                    # the original, uncompressed messages.
                    logger.warning(
                        "TTC compression failed (%s); sending uncompressed.", exc
                    )
            response = original_create(*args, **kwargs)

            if web_search:
                response = _handle_search_loop_sync(
                    response,
                    kwargs,
                    original_create,
                    sync_ttc,
                    model,
                    stats,
                    None,
                    role_aggr,
                    system_aggr,
                    strip_server_tool_results,
                    _effective_max_uses(web_search_max_uses, native_max_uses),
                )

            return response

        client.messages.create = sync_create

    client.compression = stats
    return client
