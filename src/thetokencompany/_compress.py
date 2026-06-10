"""Shared message compression logic for SDK wrappers."""

from __future__ import annotations

import asyncio
from typing import Any

from thetokencompany._async_client import AsyncTheTokenCompany
from thetokencompany._client import TheTokenCompany
from thetokencompany._types import CompressionStats, CompressResponse

DEFAULT_AGGRESSIVENESS = 0.2
DEFAULT_ROLES = ("user", "system", "tool")
_SERVER_TOOL_BLOCK_TYPES = frozenset({"web_search_tool_result", "server_tool_use"})

Aggressiveness = float | dict[str, float]


class _StatsTTC:
    """Wraps TheTokenCompany to collect compression analytics."""

    def __init__(self, inner: TheTokenCompany, stats: CompressionStats) -> None:
        self._inner = inner
        self._stats = stats

    def compress(self, text: str, *, model: str, aggressiveness: float) -> CompressResponse:
        result = self._inner.compress(text, model=model, aggressiveness=aggressiveness)
        self._stats._record(result)
        return result


class _AsyncStatsTTC:
    """Wraps AsyncTheTokenCompany to collect compression analytics."""

    def __init__(self, inner: AsyncTheTokenCompany, stats: CompressionStats) -> None:
        self._inner = inner
        self._stats = stats

    async def compress(self, text: str, *, model: str, aggressiveness: float) -> CompressResponse:
        result = await self._inner.compress(text, model=model, aggressiveness=aggressiveness)
        self._stats._record(result)
        return result


def _resolve_aggressiveness(aggressiveness: Aggressiveness) -> dict[str, float]:
    if isinstance(aggressiveness, dict):
        return aggressiveness
    return {role: aggressiveness for role in DEFAULT_ROLES}


# ---------------------------------------------------------------------------
# Text / block helpers
# ---------------------------------------------------------------------------


def compress_text(
    ttc: TheTokenCompany, text: str, model: str, aggressiveness: float
) -> str:
    if not text or not text.strip():
        return text
    return ttc.compress(text, model=model, aggressiveness=aggressiveness).output


async def compress_text_async(
    ttc: AsyncTheTokenCompany, text: str, model: str, aggressiveness: float
) -> str:
    if not text or not text.strip():
        return text
    result = await ttc.compress(text, model=model, aggressiveness=aggressiveness)
    return result.output


def _compress_text_blocks(
    ttc: TheTokenCompany, blocks: list[Any], model: str, aggressiveness: float
) -> list[Any]:
    result = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            result.append({**block, "text": compress_text(ttc, text, model, aggressiveness)})
        else:
            result.append(block)
    return result


async def _compress_text_blocks_async(
    ttc: AsyncTheTokenCompany, blocks: list[Any], model: str, aggressiveness: float
) -> list[Any]:
    async def _do(block: Any) -> Any:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            return {**block, "text": await compress_text_async(ttc, text, model, aggressiveness)}
        return block

    return list(await asyncio.gather(*[_do(b) for b in blocks]))


# ---------------------------------------------------------------------------
# OpenAI message compression
# ---------------------------------------------------------------------------

_OPENAI_TOOL_ROLES = frozenset({"tool", "function"})


def compress_openai_messages(
    ttc: TheTokenCompany,
    messages: list[dict[str, Any]],
    model: str,
    role_aggr: dict[str, float],
) -> list[dict[str, Any]]:
    return [_compress_openai_msg(ttc, m, model, role_aggr) for m in messages]


async def compress_openai_messages_async(
    ttc: AsyncTheTokenCompany,
    messages: list[dict[str, Any]],
    model: str,
    role_aggr: dict[str, float],
) -> list[dict[str, Any]]:
    tasks = [_compress_openai_msg_async(ttc, m, model, role_aggr) for m in messages]
    return list(await asyncio.gather(*tasks))


def _openai_aggr(role: str, role_aggr: dict[str, float]) -> float | None:
    if role in _OPENAI_TOOL_ROLES:
        return role_aggr.get("tool")
    return role_aggr.get(role)


def _compress_openai_msg(
    ttc: TheTokenCompany,
    message: dict[str, Any],
    model: str,
    role_aggr: dict[str, float],
) -> dict[str, Any]:
    aggr = _openai_aggr(message.get("role", ""), role_aggr)
    if aggr is None:
        return message

    content = message.get("content")
    if content is None:
        return message
    if isinstance(content, str):
        return {**message, "content": compress_text(ttc, content, model, aggr)}
    if isinstance(content, list):
        return {**message, "content": _compress_text_blocks(ttc, content, model, aggr)}
    return message


async def _compress_openai_msg_async(
    ttc: AsyncTheTokenCompany,
    message: dict[str, Any],
    model: str,
    role_aggr: dict[str, float],
) -> dict[str, Any]:
    aggr = _openai_aggr(message.get("role", ""), role_aggr)
    if aggr is None:
        return message

    content = message.get("content")
    if content is None:
        return message
    if isinstance(content, str):
        return {**message, "content": await compress_text_async(ttc, content, model, aggr)}
    if isinstance(content, list):
        return {**message, "content": await _compress_text_blocks_async(ttc, content, model, aggr)}
    return message


# ---------------------------------------------------------------------------
# Anthropic message compression
# ---------------------------------------------------------------------------


def compress_anthropic_messages(
    ttc: TheTokenCompany,
    messages: list[dict[str, Any]],
    model: str,
    role_aggr: dict[str, float],
    *,
    strip_server_tool_results: bool = False,
    max_search_results: int | None = None,
) -> list[dict[str, Any]]:
    return [
        _compress_anthropic_msg(
            ttc, m, model, role_aggr,
            strip_server_tool_results=strip_server_tool_results,
            max_search_results=max_search_results,
        )
        for m in messages
    ]


async def compress_anthropic_messages_async(
    ttc: AsyncTheTokenCompany,
    messages: list[dict[str, Any]],
    model: str,
    role_aggr: dict[str, float],
    *,
    strip_server_tool_results: bool = False,
    max_search_results: int | None = None,
) -> list[dict[str, Any]]:
    tasks = [
        _compress_anthropic_msg_async(
            ttc, m, model, role_aggr,
            strip_server_tool_results=strip_server_tool_results,
            max_search_results=max_search_results,
        )
        for m in messages
    ]
    return list(await asyncio.gather(*tasks))


def _compress_anthropic_msg(
    ttc: TheTokenCompany,
    message: dict[str, Any],
    model: str,
    role_aggr: dict[str, float],
    *,
    strip_server_tool_results: bool = False,
    max_search_results: int | None = None,
) -> dict[str, Any]:
    role = message.get("role", "")

    if role == "assistant":
        assistant_aggr = role_aggr.get("assistant")
        needs_processing = (
            assistant_aggr is not None
            or strip_server_tool_results
            or max_search_results is not None
        )
        content = message.get("content")
        if content is None or not needs_processing:
            return message
        if isinstance(content, str):
            if assistant_aggr is not None:
                return {**message, "content": compress_text(ttc, content, model, assistant_aggr)}
            return message
        if isinstance(content, list):
            blocks = _compress_assistant_blocks(
                ttc, content, model, assistant_aggr,
                strip_server_tool_results,
            )
            return {**message, "content": blocks}
        return message

    if role != "user":
        return message

    user_aggr = role_aggr.get("user")
    tool_aggr = role_aggr.get("tool")
    if user_aggr is None and tool_aggr is None:
        return message

    content = message.get("content")
    if content is None:
        return message
    if isinstance(content, str):
        if user_aggr is not None:
            return {**message, "content": compress_text(ttc, content, model, user_aggr)}
        return message
    if isinstance(content, list):
        blocks = _compress_anthropic_blocks(ttc, content, model, user_aggr, tool_aggr)
        return {**message, "content": blocks}
    return message


async def _compress_anthropic_msg_async(
    ttc: AsyncTheTokenCompany,
    message: dict[str, Any],
    model: str,
    role_aggr: dict[str, float],
    *,
    strip_server_tool_results: bool = False,
    max_search_results: int | None = None,
) -> dict[str, Any]:
    role = message.get("role", "")

    if role == "assistant":
        assistant_aggr = role_aggr.get("assistant")
        needs_processing = (
            assistant_aggr is not None
            or strip_server_tool_results
            or max_search_results is not None
        )
        content = message.get("content")
        if content is None or not needs_processing:
            return message
        if isinstance(content, str):
            if assistant_aggr is not None:
                return {**message, "content": await compress_text_async(ttc, content, model, assistant_aggr)}
            return message
        if isinstance(content, list):
            blocks = await _compress_assistant_blocks_async(
                ttc, content, model, assistant_aggr,
                strip_server_tool_results,
            )
            return {**message, "content": blocks}
        return message

    if role != "user":
        return message

    user_aggr = role_aggr.get("user")
    tool_aggr = role_aggr.get("tool")
    if user_aggr is None and tool_aggr is None:
        return message

    content = message.get("content")
    if content is None:
        return message
    if isinstance(content, str):
        if user_aggr is not None:
            return {**message, "content": await compress_text_async(ttc, content, model, user_aggr)}
        return message
    if isinstance(content, list):
        blocks = await _compress_anthropic_blocks_async(ttc, content, model, user_aggr, tool_aggr)
        return {**message, "content": blocks}
    return message


def _compress_assistant_blocks(
    ttc: TheTokenCompany,
    blocks: list[Any],
    model: str,
    assistant_aggr: float | None,
    strip_server_tool_results: bool,
) -> list[Any]:
    result = []
    for block in blocks:
        if not isinstance(block, dict):
            result.append(block)
            continue
        btype = block.get("type")
        if strip_server_tool_results and btype in _SERVER_TOOL_BLOCK_TYPES:
            continue
        if btype == "text" and assistant_aggr is not None:
            text = block.get("text", "")
            result.append({**block, "text": compress_text(ttc, text, model, assistant_aggr)})
        else:
            result.append(block)
    if not result:
        return blocks
    return result


async def _compress_assistant_blocks_async(
    ttc: AsyncTheTokenCompany,
    blocks: list[Any],
    model: str,
    assistant_aggr: float | None,
    strip_server_tool_results: bool,
) -> list[Any]:
    async def _do(block: Any) -> Any | None:
        if not isinstance(block, dict):
            return block
        btype = block.get("type")
        if strip_server_tool_results and btype in _SERVER_TOOL_BLOCK_TYPES:
            return None
        if btype == "text" and assistant_aggr is not None:
            text = block.get("text", "")
            return {**block, "text": await compress_text_async(ttc, text, model, assistant_aggr)}
        return block

    results = await asyncio.gather(*[_do(b) for b in blocks])
    filtered = [r for r in results if r is not None]
    if not filtered:
        return blocks
    return filtered


def _compress_anthropic_blocks(
    ttc: TheTokenCompany,
    blocks: list[Any],
    model: str,
    user_aggr: float | None,
    tool_aggr: float | None,
) -> list[Any]:
    result = []
    for block in blocks:
        if not isinstance(block, dict):
            result.append(block)
            continue

        btype = block.get("type")
        if btype == "text" and user_aggr is not None:
            text = block.get("text", "")
            result.append({**block, "text": compress_text(ttc, text, model, user_aggr)})
        elif btype == "tool_result" and tool_aggr is not None:
            result.append(_compress_tool_result(ttc, block, model, tool_aggr))
        else:
            result.append(block)
    return result


async def _compress_anthropic_blocks_async(
    ttc: AsyncTheTokenCompany,
    blocks: list[Any],
    model: str,
    user_aggr: float | None,
    tool_aggr: float | None,
) -> list[Any]:
    async def _do(block: Any) -> Any:
        if not isinstance(block, dict):
            return block
        btype = block.get("type")
        if btype == "text" and user_aggr is not None:
            text = block.get("text", "")
            return {**block, "text": await compress_text_async(ttc, text, model, user_aggr)}
        if btype == "tool_result" and tool_aggr is not None:
            return await _compress_tool_result_async(ttc, block, model, tool_aggr)
        return block

    return list(await asyncio.gather(*[_do(b) for b in blocks]))


def _compress_tool_result(
    ttc: TheTokenCompany,
    block: dict[str, Any],
    model: str,
    aggressiveness: float,
) -> dict[str, Any]:
    content = block.get("content")
    if isinstance(content, str):
        return {**block, "content": compress_text(ttc, content, model, aggressiveness)}
    if isinstance(content, list):
        return {**block, "content": _compress_text_blocks(ttc, content, model, aggressiveness)}
    return block


async def _compress_tool_result_async(
    ttc: AsyncTheTokenCompany,
    block: dict[str, Any],
    model: str,
    aggressiveness: float,
) -> dict[str, Any]:
    content = block.get("content")
    if isinstance(content, str):
        return {**block, "content": await compress_text_async(ttc, content, model, aggressiveness)}
    if isinstance(content, list):
        compressed = await _compress_text_blocks_async(ttc, content, model, aggressiveness)
        return {**block, "content": compressed}
    return block
