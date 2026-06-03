from __future__ import annotations

import json
from typing import Any

import httpx

from thetokencompany._client import _build_payload, _parse_error
from thetokencompany._constants import BASE_URL, DEFAULT_TIMEOUT
from thetokencompany._types import CompressResponse


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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._gzip = gzip
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if gzip:
            headers["Content-Encoding"] = "gzip"
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
    ) -> CompressResponse:
        """Compress *text* for cheaper / faster LLM inference.

        Args:
            text: The prompt or content to compress.
            model: Compression model (``bear-1``, ``bear-1.1``, ``bear-1.2``, ``bear-2``).
            aggressiveness: 0.0 (lightest) to 1.0 (most aggressive). Default 0.2.

        Returns:
            A :class:`CompressResponse` with the compressed output and token metrics.
        """
        payload = _build_payload(text, model, aggressiveness)
        response = await self._client.post(
            f"{self._base_url}/v1/compress",
            content=self._encode(payload),
        )
        if not response.is_success:
            raise _parse_error(response)
        return CompressResponse.from_dict(response.json())

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncTheTokenCompany:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
