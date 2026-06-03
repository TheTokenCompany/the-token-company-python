from __future__ import annotations

from typing import Any

import pytest

from thetokencompany import AsyncTheTokenCompany, TheTokenCompany

API_KEY = "ttc-test-key"
BASE_URL = "https://api.test.thetokencompany.com"


def make_compress_response(
    output: str = "compressed text",
    output_tokens: int = 10,
    original_input_tokens: int = 50,
) -> dict[str, Any]:
    return {
        "output": output,
        "output_tokens": output_tokens,
        "original_input_tokens": original_input_tokens,
    }


@pytest.fixture
def client() -> TheTokenCompany:
    c = TheTokenCompany(api_key=API_KEY, base_url=BASE_URL)
    yield c
    c.close()


@pytest.fixture
def async_client() -> AsyncTheTokenCompany:
    return AsyncTheTokenCompany(api_key=API_KEY, base_url=BASE_URL)
