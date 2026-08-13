"""Regression tests: a caller-supplied httpx client must never be mutated with
TTC credentials, so the bearer key cannot leak to other origins the caller
reuses that client for. See the security report on shared-client key disclosure.
"""

from __future__ import annotations

import httpx
import pytest

from thetokencompany import AsyncTheTokenCompany, TheTokenCompany

API_KEY = "ttc-test-key"
CALLER_TOKEN = "Bearer caller-placeholder-token"
COMPRESS_BODY = {"output": "x", "output_tokens": 1, "original_input_tokens": 2}


def _recording_transport(seen: dict[str, httpx.Headers]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen[str(request.url)] = request.headers
        return httpx.Response(200, json=COMPRESS_BODY)

    return httpx.MockTransport(handler)


class TestSuppliedClientNotMutated:
    def test_sync_does_not_leak_key_to_other_origin(self) -> None:
        seen: dict[str, httpx.Headers] = {}
        shared = httpx.Client(
            transport=_recording_transport(seen),
            headers={"Authorization": CALLER_TOKEN},
        )
        client = TheTokenCompany(api_key=API_KEY, http_client=shared)

        # Constructing the SDK must not touch the client's global defaults.
        assert shared.headers["authorization"] == CALLER_TOKEN

        # A TTC call still authenticates against the TTC endpoint...
        client.compress("hello world")
        ttc = seen["https://api.thetokencompany.com/v1/compress"]
        assert ttc["authorization"] == f"Bearer {API_KEY}"
        assert ttc["content-encoding"] == "gzip"

        # ...but reusing the client for another origin carries only the caller's
        # own header — never the TTC bearer key.
        shared.get("https://third-party.example/whoami")
        third = seen["https://third-party.example/whoami"]
        assert third["authorization"] == CALLER_TOKEN
        assert f"Bearer {API_KEY}" not in third.get("authorization", "")
        assert "content-encoding" not in third

        shared.close()

    @pytest.mark.asyncio
    async def test_async_does_not_leak_key_to_other_origin(self) -> None:
        seen: dict[str, httpx.Headers] = {}
        shared = httpx.AsyncClient(
            transport=_recording_transport(seen),
            headers={"Authorization": CALLER_TOKEN},
        )
        client = AsyncTheTokenCompany(api_key=API_KEY, http_client=shared)

        assert shared.headers["authorization"] == CALLER_TOKEN

        await client.compress("hello world")
        ttc = seen["https://api.thetokencompany.com/v1/compress"]
        assert ttc["authorization"] == f"Bearer {API_KEY}"
        assert ttc["content-encoding"] == "gzip"

        await shared.get("https://third-party.example/whoami")
        third = seen["https://third-party.example/whoami"]
        assert third["authorization"] == CALLER_TOKEN
        assert "content-encoding" not in third

        await shared.aclose()
