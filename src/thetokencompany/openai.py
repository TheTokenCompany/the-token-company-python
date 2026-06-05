"""Auto-compress messages for OpenAI, OpenRouter, and compatible clients."""

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
    _resolve_aggressiveness,
    compress_openai_messages,
    compress_openai_messages_async,
)
from thetokencompany._constants import BEAR_2
from thetokencompany._types import CompressionStats


def with_compression(
    client: Any,
    *,
    compression_api_key: str,
    model: str = BEAR_2,
    aggressiveness: Aggressiveness = DEFAULT_AGGRESSIVENESS,
    app_id: str | None = None,
) -> Any:
    """Wrap an OpenAI-compatible client to auto-compress messages.

    Works with ``openai.OpenAI``, ``openai.AsyncOpenAI``, and any
    OpenAI-compatible client (OpenRouter, Together, etc.).

    Compression stats are available on ``client.compression``::

        client = with_compression(OpenAI(), compression_api_key="ttc-...")
        client.chat.completions.create(...)
        print(client.compression.total_tokens_saved)

    Assistant messages are never compressed to preserve the provider's KV cache.
    """
    role_aggr = _resolve_aggressiveness(aggressiveness)
    stats = CompressionStats()
    original_create = client.chat.completions.create

    if inspect.iscoroutinefunction(original_create):
        tracker = _AsyncAnalyticsTTC(AsyncTheTokenCompany(api_key=compression_api_key, app_id=app_id), stats)

        @functools.wraps(original_create)
        async def async_create(*args: Any, **kwargs: Any) -> Any:
            if "messages" in kwargs:
                stats._start_turn()
                kwargs["messages"] = await compress_openai_messages_async(
                    tracker, kwargs["messages"], model, role_aggr  # type: ignore[arg-type]
                )
                stats._end_turn()
            return await original_create(*args, **kwargs)

        client.chat.completions.create = async_create
    else:
        tracker = _AnalyticsTTC(TheTokenCompany(api_key=compression_api_key, app_id=app_id), stats)  # type: ignore[assignment]

        @functools.wraps(original_create)
        def sync_create(*args: Any, **kwargs: Any) -> Any:
            if "messages" in kwargs:
                stats._start_turn()
                kwargs["messages"] = compress_openai_messages(
                    tracker, kwargs["messages"], model, role_aggr  # type: ignore[arg-type]
                )
                stats._end_turn()
            return original_create(*args, **kwargs)

        client.chat.completions.create = sync_create

    client.compression = stats
    return client
