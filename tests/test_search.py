from __future__ import annotations

import json

import httpx
import pytest
import respx

from thetokencompany import (
    AsyncTheTokenCompany,
    SearchResponse,
    SearchResult,
    TheTokenCompany,
)

from .conftest import BASE_URL


def _make_search_response(
    *,
    results: list[dict] | None = None,
    query: str = "test query",
    search_time: float = 0.5,
    original_input_tokens: int = 1000,
    output_tokens: int = 700,
) -> dict:
    if results is None:
        results = [
            {
                "url": "https://example.com",
                "title": "Example",
                "content": "compressed text",
                "score": 0.95,
            }
        ]
    return {
        "results": results,
        "query": query,
        "search_time": search_time,
        "original_input_tokens": original_input_tokens,
        "output_tokens": output_tokens,
    }


class TestSearch:
    @respx.mock
    def test_basic_search(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=_make_search_response())
        )
        result = client.search("test query")
        assert isinstance(result, SearchResponse)
        assert len(result.results) == 1
        assert result.results[0].url == "https://example.com"
        assert result.results[0].title == "Example"
        assert result.results[0].content == "compressed text"
        assert result.results[0].score == 0.95
        assert result.query == "test query"
        assert result.tokens_saved == 300

    @respx.mock
    def test_search_empty_results(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/search").mock(
            return_value=httpx.Response(
                200,
                json=_make_search_response(
                    results=[],
                    query="nothing here",
                    original_input_tokens=0,
                    output_tokens=0,
                ),
            )
        )
        result = client.search("nothing here")
        assert result.results == []
        assert result.query == "nothing here"
        assert result.tokens_saved == 0

    @respx.mock
    def test_search_sends_payload(self, client: TheTokenCompany) -> None:
        import gzip

        route = respx.post(f"{BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=_make_search_response())
        )
        client.search(
            "test query",
            max_results=10,
            search_depth="advanced",
            include_raw_content=True,
            model="bear-2",
            aggressiveness=0.5,
        )
        body = json.loads(gzip.decompress(route.calls[0].request.content))
        assert body["query"] == "test query"
        assert body["max_results"] == 10
        assert body["search_depth"] == "advanced"
        assert body["include_raw_content"] is True
        assert body["model"] == "bear-2"
        assert body["compression_settings"]["aggressiveness"] == 0.5

    @respx.mock
    def test_search_custom_base_url(self) -> None:
        staging_url = "https://staging.api.thetokencompany.com"
        respx.post(f"{staging_url}/v1/search").mock(
            return_value=httpx.Response(
                200,
                json=_make_search_response(results=[], query="test"),
            )
        )
        client = TheTokenCompany(api_key="ttc-test", base_url=staging_url, gzip=False)
        result = client.search("test")
        assert result.query == "test"
        assert isinstance(result, SearchResponse)
        client.close()

    @respx.mock
    def test_search_sends_app_id(self) -> None:
        import gzip

        route = respx.post(f"{BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=_make_search_response())
        )
        client = TheTokenCompany(api_key="ttc-test", base_url=BASE_URL, app_id="my-app")
        client.search("test query")
        body = json.loads(gzip.decompress(route.calls[0].request.content))
        assert body["app_id"] == "my-app"
        client.close()

    @respx.mock
    def test_search_override_app_id(self) -> None:
        import gzip

        route = respx.post(f"{BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=_make_search_response())
        )
        client = TheTokenCompany(api_key="ttc-test", base_url=BASE_URL, app_id="default-app")
        client.search("test query", app_id="override-app")
        body = json.loads(gzip.decompress(route.calls[0].request.content))
        assert body["app_id"] == "override-app"
        client.close()

    @respx.mock
    def test_search_error_handling(self, client: TheTokenCompany) -> None:
        from thetokencompany import AuthenticationError

        respx.post(f"{BASE_URL}/v1/search").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            client.search("test query")

    @respx.mock
    def test_search_no_score(self, client: TheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/search").mock(
            return_value=httpx.Response(
                200,
                json=_make_search_response(
                    results=[
                        {
                            "url": "https://example.com",
                            "title": "No Score",
                            "content": "text",
                        }
                    ]
                ),
            )
        )
        result = client.search("test")
        assert result.results[0].score is None


class TestSearchResponse:
    def test_from_dict(self) -> None:
        data = _make_search_response()
        resp = SearchResponse.from_dict(data)
        assert len(resp.results) == 1
        assert isinstance(resp.results[0], SearchResult)
        assert resp.query == "test query"
        assert resp.search_time == 0.5

    def test_tokens_saved(self) -> None:
        resp = SearchResponse(
            results=[],
            query="q",
            original_input_tokens=1000,
            output_tokens=600,
        )
        assert resp.tokens_saved == 400

    def test_tokens_saved_zero(self) -> None:
        resp = SearchResponse(results=[], query="q")
        assert resp.tokens_saved == 0


class TestAsyncSearch:
    @pytest.mark.asyncio
    @respx.mock
    async def test_basic_async_search(self, async_client: AsyncTheTokenCompany) -> None:
        respx.post(f"{BASE_URL}/v1/search").mock(
            return_value=httpx.Response(200, json=_make_search_response())
        )
        result = await async_client.search("test query")
        assert isinstance(result, SearchResponse)
        assert len(result.results) == 1
        assert result.results[0].url == "https://example.com"
        assert result.tokens_saved == 300
