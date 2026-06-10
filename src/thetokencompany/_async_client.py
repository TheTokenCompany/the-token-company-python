from __future__ import annotations

import json
from typing import Any

import httpx

from thetokencompany._client import _build_payload, _parse_error
from thetokencompany._constants import BASE_URL, DEFAULT_TIMEOUT
from thetokencompany._types import CompressResponse, SearchResponse


class AsyncTheTokenCompany:
    """Async client for The Token Company compression API.

    Usage::

        from thetokencompany import AsyncTheTokenCompany

        async with AsyncTheTokenCompany(api_key="ttc-...") as client:
            result = await client.compress("Your long prompt text...", model="bear-2")
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
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._gzip = gzip
        self._app_id = app_id
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if gzip:
            headers["Content-Encoding"] = "gzip"
        if http_client is not None:
            self._client = http_client
            self._client.headers.update(headers)
        else:
            self._client = httpx.AsyncClient(headers=headers, timeout=timeout)

    def _encode(self, payload: dict[str, Any]) -> bytes:
        raw = json.dumps(payload).encode()
        if self._gzip:
            import gzip as _gzip

            return _gzip.compress(raw)
        return raw

    async def compress(
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
        response = await self._client.post(
            f"{self._base_url}/v1/compress",
            content=self._encode(payload),
        )
        if not response.is_success:
            raise _parse_error(response)
        return CompressResponse.from_dict(response.json())

    async def search(
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
        response = await self._client.post(
            f"{self._base_url}/v1/search",
            content=self._encode(payload),
        )
        if not response.is_success:
            raise _parse_error(response)
        return SearchResponse.from_dict(response.json())

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncTheTokenCompany:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
