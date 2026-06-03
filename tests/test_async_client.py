from __future__ import annotations

import httpx
import pytest
import respx

from thetokencompany import AsyncTheTokenCompany, AuthenticationError, CompressResponse

from .conftest import BASE_URL, make_compress_response


class TestAsyncCompress:
    @respx.mock
    async def test_basic(self, async_client: AsyncTheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        result = await async_client.compress("hello world", model="bear-1.2")
        assert isinstance(result, CompressResponse)
        assert result.output == "compressed text"
        await async_client.close()

    @respx.mock
    async def test_context_manager(self) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        async with AsyncTheTokenCompany(api_key="k", base_url=BASE_URL) as client:
            result = await client.compress("hello")
        assert result.output == "compressed text"

    @respx.mock
    async def test_error(self, async_client: AsyncTheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(401, json={"detail": "Bad key"})
        )
        with pytest.raises(AuthenticationError):
            await async_client.compress("hello")
        await async_client.close()
