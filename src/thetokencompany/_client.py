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
from thetokencompany._types import CompressResponse

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
) -> dict[str, Any]:
    if not text or not text.strip():
        raise InvalidRequestError("text cannot be empty")
    if not 0.0 <= aggressiveness <= 1.0:
        raise InvalidRequestError("aggressiveness must be between 0.0 and 1.0")
    return {
        "model": model,
        "input": text,
        "compression_settings": {"aggressiveness": aggressiveness},
    }


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
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._gzip = gzip
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if gzip:
            headers["Content-Encoding"] = "gzip"
        self._client = httpx.Client(headers=headers, timeout=timeout)

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
        response = self._client.post(
            f"{self._base_url}/v1/compress",
            content=self._encode(payload),
        )
        if not response.is_success:
            raise _parse_error(response)
        return CompressResponse.from_dict(response.json())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TheTokenCompany:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
