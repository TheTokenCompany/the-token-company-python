"""The Token Company Python SDK — compress LLM prompts to reduce costs and latency."""

from thetokencompany._async_client import AsyncTheTokenCompany
from thetokencompany._client import TheTokenCompany
from thetokencompany._constants import BEAR_1, BEAR_1_1, BEAR_1_2, BEAR_2
from thetokencompany._exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    PaymentRequiredError,
    RateLimitError,
    RequestTooLargeError,
    TheTokenCompanyError,
)
from thetokencompany._types import (
    ChatCompressResponse,
    CompressionStats,
    CompressResponse,
    SearchResponse,
    SearchResult,
    TurnStats,
    protect,
)

__version__ = "0.4.0"

__all__ = [
    "AsyncTheTokenCompany",
    "TheTokenCompany",
    "ChatCompressResponse",
    "CompressResponse",
    "CompressionStats",
    "SearchResponse",
    "SearchResult",
    "TurnStats",
    "protect",
    "BEAR_1",
    "BEAR_1_1",
    "BEAR_1_2",
    "BEAR_2",
    "TheTokenCompanyError",
    "APIError",
    "AuthenticationError",
    "InvalidRequestError",
    "PaymentRequiredError",
    "RateLimitError",
    "RequestTooLargeError",
]
