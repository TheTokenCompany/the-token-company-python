"""Async compression with multiple texts in parallel."""

import asyncio

from thetokencompany import AsyncTheTokenCompany

TEXTS = [
    "First document with verbose content that can be compressed...",
    "Second document with even more filler words to remove...",
    "Third document ready for compression processing...",
]


async def main() -> None:
    async with AsyncTheTokenCompany(api_key="your-api-key") as client:
        results = await asyncio.gather(
            *(client.compress(text, model="bear-1.2") for text in TEXTS)
        )
        total_saved = sum(r.tokens_saved for r in results)
        for i, r in enumerate(results, 1):
            print(f"Doc {i}: {r.tokens_saved} tokens saved ({r.compression_ratio:.1f}x)")
        print(f"Total saved: {total_saved}")


if __name__ == "__main__":
    asyncio.run(main())
