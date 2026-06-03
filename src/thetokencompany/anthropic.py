"""Auto-compress messages for Anthropic clients."""

from __future__ import annotations

import functools
import inspect
from typing import Any

from thetokencompany._async_client import AsyncTheTokenCompany
from thetokencompany._client import TheTokenCompany
from thetokencompany._compress import (
    DEFAULT_AGGRESSIVENESS,
    Aggressiveness,
    _AnalyticsTTC,
    _AsyncAnalyticsTTC,
    _compress_text_blocks,
    _compress_text_blocks_async,
    _resolve_aggressiveness,
    compress_anthropic_messages,
    compress_anthropic_messages_async,
    compress_text,
    compress_text_async,
)
from thetokencompany._constants import BEAR_2
from thetokencompany._types import CompressionStats


def with_compression(
    client: Any,
    *,
    compression_api_key: str,
    model: str = BEAR_2,
    aggressiveness: Aggressiveness = DEFAULT_AGGRESSIVENESS,
) -> Any:
    """Wrap an Anthropic client to auto-compress messages.

    Compresses the ``system`` parameter and all non-assistant messages.

    Compression stats are available on ``client.compression``::

        client = with_compression(Anthropic(), compression_api_key="ttc-...")
        client.messages.create(...)
        print(client.compression.total_tokens_saved)

    Assistant messages are never compressed to preserve the provider's KV cache.
    """
    role_aggr = _resolve_aggressiveness(aggressiveness)
    system_aggr = role_aggr.get("system")
    stats = CompressionStats()
    original_create = client.messages.create

    if inspect.iscoroutinefunction(original_create):
        tracker = _AsyncAnalyticsTTC(AsyncTheTokenCompany(api_key=compression_api_key), stats)

        @functools.wraps(original_create)
        async def async_create(*args: Any, **kwargs: Any) -> Any:
            stats._start_turn()
            if "messages" in kwargs:
                kwargs["messages"] = await compress_anthropic_messages_async(
                    tracker, kwargs["messages"], model, role_aggr  # type: ignore[arg-type]
                )
            if system_aggr is not None and "system" in kwargs:
                system = kwargs["system"]
                if isinstance(system, str):
                    kwargs["system"] = await compress_text_async(
                        tracker, system, model, system_aggr  # type: ignore[arg-type]
                    )
                elif isinstance(system, list):
                    kwargs["system"] = await _compress_text_blocks_async(
                        tracker, system, model, system_aggr  # type: ignore[arg-type]
                    )
            stats._end_turn()
            return await original_create(*args, **kwargs)

        client.messages.create = async_create
    else:
        tracker = _AnalyticsTTC(TheTokenCompany(api_key=compression_api_key), stats)  # type: ignore[assignment]

        @functools.wraps(original_create)
        def sync_create(*args: Any, **kwargs: Any) -> Any:
            stats._start_turn()
            if "messages" in kwargs:
                kwargs["messages"] = compress_anthropic_messages(
                    tracker, kwargs["messages"], model, role_aggr  # type: ignore[arg-type]
                )
            if system_aggr is not None and "system" in kwargs:
                system = kwargs["system"]
                if isinstance(system, str):
                    kwargs["system"] = compress_text(
                        tracker, system, model, system_aggr  # type: ignore[arg-type]
                    )
                elif isinstance(system, list):
                    kwargs["system"] = _compress_text_blocks(
                        tracker, system, model, system_aggr  # type: ignore[arg-type]
                    )
            stats._end_turn()
            return original_create(*args, **kwargs)

        client.messages.create = sync_create

    client.compression = stats
    return client
