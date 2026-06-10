from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CompressResponse:
    """Result of a compression request."""

    output: str
    output_tokens: int
    input_tokens: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressResponse:
        return cls(
            output=data["output"],
            output_tokens=data["output_tokens"],
            input_tokens=data["original_input_tokens"],
        )

    @property
    def tokens_saved(self) -> int:
        return self.input_tokens - self.output_tokens

    @property
    def compression_ratio(self) -> float:
        if self.output_tokens == 0:
            return 0.0
        return self.input_tokens / self.output_tokens



@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single compressed web search result."""

    url: str
    title: str
    content: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Result of a search request."""

    results: list[SearchResult]
    query: str
    search_time: float = 0
    original_input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResponse:
        results = [
            SearchResult(
                url=r["url"],
                title=r["title"],
                content=r["content"],
                score=r.get("score"),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            query=data.get("query", ""),
            search_time=data.get("search_time", 0),
            original_input_tokens=data.get("original_input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )

    @property
    def tokens_saved(self) -> int:
        return self.original_input_tokens - self.output_tokens


@dataclass(frozen=True, slots=True)
class TurnStats:
    """Stats from a single create() call."""

    input_tokens: int
    output_tokens: int
    tokens_saved: int
    messages_compressed: int
    timestamp: float

    @property
    def ratio(self) -> float:
        if self.output_tokens == 0:
            return 0.0
        return self.input_tokens / self.output_tokens


@dataclass
class CompressionStats:
    """Aggregate compression stats across all create() calls."""

    history: list[TurnStats] = field(default_factory=list)
    _accumulator: list[CompressResponse] = field(default_factory=list, repr=False)

    def _start_turn(self) -> None:
        self._accumulator = []

    def _record(self, response: CompressResponse) -> None:
        self._accumulator.append(response)

    def _end_turn(self) -> None:
        if self._accumulator:
            input_tokens = sum(r.input_tokens for r in self._accumulator)
            output_tokens = sum(r.output_tokens for r in self._accumulator)
            self.history.append(
                TurnStats(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=input_tokens - output_tokens,
                    messages_compressed=len(self._accumulator),
                    timestamp=time.time(),
                )
            )
        self._accumulator = []

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.history)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.history)

    @property
    def total_tokens_saved(self) -> int:
        return sum(t.tokens_saved for t in self.history)

    @property
    def calls(self) -> int:
        return len(self.history)

    @property
    def ratio(self) -> float:
        if self.total_output_tokens == 0:
            return 0.0
        return self.total_input_tokens / self.total_output_tokens


def protect(text: str) -> str:
    """Wrap *text* in ``<ttc_safe>`` tags so it is excluded from compression."""
    return f"<ttc_safe>{text}</ttc_safe>"
