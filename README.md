# The Token Company Python SDK

Compress LLM prompts to reduce costs and latency. 100K tokens compressed in ~85ms.

[![PyPI version](https://img.shields.io/pypi/v/the-token-company)](https://pypi.org/project/the-token-company/)
[![PyPI downloads](https://img.shields.io/pypi/dm/the-token-company)](https://pypi.org/project/the-token-company/)
[![Python versions](https://img.shields.io/pypi/pyversions/the-token-company)](https://pypi.org/project/the-token-company/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/TheTokenCompany/the-token-company-python/blob/main/LICENSE)

[Docs](https://thetokencompany.com/docs) · [Website](https://thetokencompany.com) · [Dashboard](https://app.thetokencompany.com) · [Node.js SDK](https://github.com/TheTokenCompany/the-token-company-node)

## Install

```bash
pip install the-token-company
```

## Quick start

```python
from thetokencompany import TheTokenCompany

client = TheTokenCompany(api_key="ttc-...")
result = client.compress("Your long prompt text here...", model="bear-2")

print(result.output)           # compressed text
print(result.tokens_saved)     # tokens removed
print(result.compression_ratio)  # e.g. 1.8
```

## SDK wrappers

Drop-in wrappers that auto-compress all non-assistant messages before sending to your LLM. Assistant messages pass through unchanged so the provider's KV cache stays warm.

### OpenAI / OpenRouter

```python
from openai import OpenAI
from thetokencompany.openai import with_compression

client = with_compression(OpenAI(), compression_api_key="ttc-...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "Summarize these results..."},
    ],
)
```

Works with `AsyncOpenAI` too — the wrapper detects async automatically.

### Anthropic

```python
from anthropic import Anthropic
from thetokencompany.anthropic import with_compression

client = with_compression(Anthropic(), compression_api_key="ttc-...")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system="You are a helpful assistant...",
    messages=[{"role": "user", "content": "Summarize these results..."}],
)
```

Both `messages` and the `system` parameter are compressed.

## Async

```python
from thetokencompany import AsyncTheTokenCompany

async with AsyncTheTokenCompany(api_key="ttc-...") as client:
    result = await client.compress("Your long prompt text...")
```

## Models

| Model      | Description            |
|------------|------------------------|
| `bear-2`   | Latest, recommended    |
| `bear-1.2` | Previous generation    |
| `bear-1.1` | Legacy                 |
| `bear-1`   | Legacy                 |

## Aggressiveness

Control compression intensity with `aggressiveness` (0.0 – 1.0, default 0.5):

```python
result = client.compress(text, model="bear-2", aggressiveness=0.8)
```

## Gzip

Enable gzip compression of request payloads for better performance on large inputs (up to 2.2x faster on 1M+ tokens):

```python
client = TheTokenCompany(api_key="ttc-...", gzip=True)
```

## Protect text from compression

Use `protect()` to wrap content in `<ttc_safe>` tags — protected text passes through unchanged:

```python
from thetokencompany import protect

prompt = f"{protect('system:')} You are a helpful assistant.\n{protect('user:')} Hello!"
result = client.compress(prompt, model="bear-2")
```

## Response

`CompressResponse` fields:

| Field              | Type    | Description                        |
|--------------------|---------|------------------------------------|
| `output`           | `str`   | Compressed text                    |
| `output_tokens`    | `int`   | Token count after compression      |
| `input_tokens`     | `int`   | Token count before compression     |
| `tokens_saved`     | `int`   | Tokens removed                     |
| `compression_ratio`| `float` | Ratio (e.g. 1.8x)                 |

## License

MIT
