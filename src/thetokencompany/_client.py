from __future__ import annotations

import json
from typing import Any

import httpx

from thetokencompany._constants import BASE_URL, DEFAULT_TIMEOUT
from thetokencompany._exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    PaymentRequiredError,
    RateLimitError,
    RequestTooLargeError,
)
from thetokencompany._types import ChatCompressResponse, CompressResponse, SearchResponse

_STATUS_MAP: dict[int, type[Exception]] = {
    401: AuthenticationError,
    402: PaymentRequiredError,
    413: RequestTooLargeError,
    429: RateLimitError,
}


def _parse_error(response: httpx.Response) -> Exception:
    try:
        body = response.json()
        detail = body.get("detail", "Unknown error")
        msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
    except Exception:
        msg = response.text or "Unknown error"

    exc_cls = _STATUS_MAP.get(response.status_code)
    if exc_cls is not None:
        return exc_cls(msg)
    if response.status_code == 400:
        return InvalidRequestError(msg)
    return APIError(f"API error ({response.status_code}): {msg}", status_code=response.status_code)


def _build_payload(
    text: str,
    model: str,
    aggressiveness: float,
    app_id: str | None = None,
) -> dict[str, Any]:
    if not text or not text.strip():
        raise InvalidRequestError("text cannot be empty")
    if not 0.0 <= aggressiveness <= 1.0:
        raise InvalidRequestError("aggressiveness must be between 0.0 and 1.0")
    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "compression_settings": {"aggressiveness": aggressiveness},
    }
    if app_id is not None:
        payload["app_id"] = app_id
    return payload


def _build_chat_payload(
    messages: list[Any],
    model: str,
    fmt: str,
    aggressiveness: float | dict[str, float],
    *,
    system: Any | None = None,
    strip_server_tool_results: bool = False,
    skip_tool_use_ids: list[str] | None = None,
    app_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "format": fmt,
        "messages": messages,
        "aggressiveness": aggressiveness,
    }
    if system is not None:
        payload["system"] = system
    if strip_server_tool_results:
        payload["strip_server_tool_results"] = True
    if skip_tool_use_ids:
        payload["skip_tool_use_ids"] = skip_tool_use_ids
    if app_id is not None:
        payload["app_id"] = app_id
    return payload


class TheTokenCompany:
    """Synchronous client for The Token Company compression API.

    Usage::

        from thetokencompany import TheTokenCompany

        client = TheTokenCompany(api_key="ttc-...")
        result = client.compress("Your long prompt text...", model="bear-2")
        print(result.output, result.tokens_saved)
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        gzip: bool = True,
        app_id: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._gzip = gzip
        self._app_id = app_id
        # TTC auth/compression headers are applied per-request (see _post), never
        # written to the client's global defaults. Mutating a caller-supplied
        # client would leak the bearer key to every other origin that client is
        # later reused for.
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if gzip:
            self._headers["Content-Encoding"] = "gzip"
        if http_client is not None:
            self._client = http_client
        else:
            # keepalive_expiry defaults to 5s in httpx — shorter than the model's
            # think-gaps between agentic search rounds, so the connection would be
            # dropped and every round would pay a fresh TLS handshake (expensive on
            # high-latency links). 300s keeps it warm across the whole run.
            self._client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=32,
                    max_connections=128,
                    keepalive_expiry=300.0,
                ),
            )

    def _encode(self, payload: dict[str, Any]) -> bytes:
        raw = json.dumps(payload).encode()
        if self._gzip:
            import gzip as _gzip

            return _gzip.compress(raw)
        return raw

    def compress(
        self,
        text: str,
        *,
        model: str = "bear-2",
        aggressiveness: float = 0.2,
        app_id: str | None = None,
    ) -> CompressResponse:
        """Compress *text* for cheaper / faster LLM inference.

        Args:
            text: The prompt or content to compress.
            model: Compression model (``bear-1``, ``bear-1.1``, ``bear-1.2``, ``bear-2``).
            aggressiveness: 0.0 (lightest) to 1.0 (most aggressive). Default 0.2.
            app_id: Optional application identifier (max 255 chars). Overrides client-level app_id.

        Returns:
            A :class:`CompressResponse` with the compressed output and token metrics.
        """
        resolved_app_id = app_id if app_id is not None else self._app_id
        payload = _build_payload(text, model, aggressiveness, app_id=resolved_app_id)
        response = self._client.post(
            f"{self._base_url}/v1/compress",
            content=self._encode(payload),
            headers=self._headers,
        )
        if not response.is_success:
            raise _parse_error(response)
        return CompressResponse.from_dict(response.json())

    def compress_chat(
        self,
        messages: list[Any],
        *,
        model: str = "bear-2",
        fmt: str = "openai",
        aggressiveness: float | dict[str, float] = 0.2,
        system: Any | None = None,
        strip_server_tool_results: bool = False,
        skip_tool_use_ids: list[str] | None = None,
        app_id: str | None = None,
    ) -> ChatCompressResponse:
        """Compress a whole chat conversation in a single request.

        Sends the entire message array to ``/v1/chat/compress`` instead of one
        request per message, so a long multi-turn conversation costs one
        round-trip and the re-sent history is served from cache server-side.

        Args:
            messages: The OpenAI/Anthropic message array.
            model: Compression model. Default ``bear-2``.
            fmt: ``"openai"`` or ``"anthropic"`` (controls block walking).
            aggressiveness: A single float, or a ``{role: float}`` map. Roles
                absent from the map are left uncompressed.
            system: Anthropic top-level ``system`` prompt (str or blocks).
            strip_server_tool_results: Drop Anthropic server tool-result blocks.
            skip_tool_use_ids: Anthropic ``tool_result`` ids to leave untouched.
            app_id: Optional application identifier. Overrides client-level app_id.

        Returns:
            A :class:`ChatCompressResponse` with the compressed ``messages``.
        """
        resolved_app_id = app_id if app_id is not None else self._app_id
        payload = _build_chat_payload(
            messages, model, fmt, aggressiveness,
            system=system,
            strip_server_tool_results=strip_server_tool_results,
            skip_tool_use_ids=skip_tool_use_ids,
            app_id=resolved_app_id,
        )
        response = self._client.post(
            f"{self._base_url}/v1/chat/compress",
            content=self._encode(payload),
            headers=self._headers,
        )
        if not response.is_success:
            raise _parse_error(response)
        return ChatCompressResponse.from_dict(response.json())

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_raw_content: bool = False,
        model: str = "bear-2",
        aggressiveness: float = 0.3,
        app_id: str | None = None,
    ) -> SearchResponse:
        """Search the web and return compressed results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return (default 5).
            search_depth: Search depth (``basic`` or ``advanced``).
            include_raw_content: Whether to include raw page content.
            model: Compression model. Default ``bear-2``.
            aggressiveness: Compression aggressiveness (0.0 to 1.0). Default 0.3.
            app_id: Optional application identifier. Overrides client-level app_id.

        Returns:
            A :class:`SearchResponse` with compressed search results and token metrics.
        """
        resolved_app_id = app_id if app_id is not None else self._app_id
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_raw_content": include_raw_content,
            "model": model,
            "compression_settings": {"aggressiveness": aggressiveness},
        }
        if resolved_app_id is not None:
            payload["app_id"] = resolved_app_id
        response = self._client.post(
            f"{self._base_url}/v1/search",
            content=self._encode(payload),
            headers=self._headers,
        )
        if not response.is_success:
            raise _parse_error(response)
        return SearchResponse.from_dict(response.json())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TheTokenCompany:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
