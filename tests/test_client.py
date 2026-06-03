from __future__ import annotations

import json

import httpx
import pytest
import respx

from thetokencompany import (
    AuthenticationError,
    CompressResponse,
    InvalidRequestError,
    PaymentRequiredError,
    RateLimitError,
    RequestTooLargeError,
    TheTokenCompany,
)

from .conftest import BASE_URL, make_compress_response


class TestCompress:
    @respx.mock
    def test_basic(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        result = client.compress("hello world", model="bear-1.2")
        assert isinstance(result, CompressResponse)
        assert result.output == "compressed text"
        assert result.input_tokens == 50
        assert result.output_tokens == 10

    @respx.mock
    def test_default_model(self, client: TheTokenCompany) -> None:
        import gzip

        route = respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        client.compress("hello world")
        body = json.loads(gzip.decompress(route.calls[0].request.content))
        assert body["model"] == "bear-2"

    @respx.mock
    def test_custom_aggressiveness(self, client: TheTokenCompany) -> None:
        import gzip

        route = respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        client.compress("hello world", aggressiveness=0.9)
        body = json.loads(gzip.decompress(route.calls[0].request.content))
        assert body["compression_settings"]["aggressiveness"] == 0.9

    @respx.mock
    def test_gzip_on_by_default(self, client: TheTokenCompany) -> None:
        import gzip

        route = respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        client.compress("hello world")
        req = route.calls[0].request
        assert req.headers["content-encoding"] == "gzip"
        body = json.loads(gzip.decompress(req.content))
        assert body["model"] == "bear-2"

    @respx.mock
    def test_gzip_disabled(self) -> None:
        no_gzip_client = TheTokenCompany(api_key="k", base_url=BASE_URL, gzip=False)
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        no_gzip_client.compress("hello world")
        req = respx.calls[0].request
        assert "content-encoding" not in req.headers
        no_gzip_client.close()

    @respx.mock
    def test_sends_auth_header(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        client.compress("hello world")
        req = respx.calls[0].request
        assert req.headers["authorization"] == "Bearer ttc-test-key"


class TestCompressResponse:
    def test_tokens_saved(self) -> None:
        r = CompressResponse(output="x", output_tokens=10, input_tokens=50)
        assert r.tokens_saved == 40

    def test_compression_ratio(self) -> None:
        r = CompressResponse(output="x", output_tokens=10, input_tokens=50)
        assert r.compression_ratio == 5.0

    def test_zero_output_tokens(self) -> None:
        r = CompressResponse(output="", output_tokens=0, input_tokens=50)
        assert r.compression_ratio == 0.0


class TestValidation:
    def test_empty_text(self, client: TheTokenCompany) -> None:
        with pytest.raises(InvalidRequestError, match="text cannot be empty"):
            client.compress("")

    def test_whitespace_text(self, client: TheTokenCompany) -> None:
        with pytest.raises(InvalidRequestError, match="text cannot be empty"):
            client.compress("   ")

    def test_aggressiveness_too_high(self, client: TheTokenCompany) -> None:
        with pytest.raises(InvalidRequestError, match="aggressiveness"):
            client.compress("hello", aggressiveness=1.5)

    def test_aggressiveness_negative(self, client: TheTokenCompany) -> None:
        with pytest.raises(InvalidRequestError, match="aggressiveness"):
            client.compress("hello", aggressiveness=-0.1)


class TestErrorHandling:
    @respx.mock
    def test_401(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client.compress("hello")

    @respx.mock
    def test_402(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(402, json={"detail": "Insufficient balance"})
        )
        with pytest.raises(PaymentRequiredError):
            client.compress("hello")

    @respx.mock
    def test_413(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(413, json={"detail": "Payload too large"})
        )
        with pytest.raises(RequestTooLargeError):
            client.compress("hello")

    @respx.mock
    def test_429(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(429, json={"detail": "Rate limit exceeded"})
        )
        with pytest.raises(RateLimitError):
            client.compress("hello")

    @respx.mock
    def test_400(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(400, json={"detail": "Bad request"})
        )
        with pytest.raises(InvalidRequestError, match="Bad request"):
            client.compress("hello")

    @respx.mock
    def test_500(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(500, json={"detail": "Internal error"})
        )
        from thetokencompany import APIError

        with pytest.raises(APIError):
            client.compress("hello")

    @respx.mock
    def test_error_with_dict_detail(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(
                401, json={"detail": {"message": "Key expired", "code": "expired"}}
            )
        )
        with pytest.raises(AuthenticationError, match="Key expired"):
            client.compress("hello")


class TestContextManager:
    @respx.mock
    def test_with_statement(self) -> None:
        respx.post(f"{BASE_URL}/v1/compress").mock(
            return_value=httpx.Response(200, json=make_compress_response())
        )
        with TheTokenCompany(api_key="k", base_url=BASE_URL) as client:
            result = client.compress("hello")
        assert result.output == "compressed text"
