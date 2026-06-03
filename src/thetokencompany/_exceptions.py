from __future__ import annotations


class TheTokenCompanyError(Exception):
    """Base exception for all SDK errors."""


class AuthenticationError(TheTokenCompanyError):
    """Invalid or missing API key."""


class InvalidRequestError(TheTokenCompanyError):
    """The request parameters were rejected by the API."""


class PaymentRequiredError(TheTokenCompanyError):
    """Insufficient balance or exceeded debt limit."""


class RequestTooLargeError(TheTokenCompanyError):
    """Request payload exceeds size limits."""


class RateLimitError(TheTokenCompanyError):
    """Too many requests."""


class APIError(TheTokenCompanyError):
    """Unexpected API error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
