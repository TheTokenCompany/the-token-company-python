"""Basic compression example."""

from thetokencompany import TheTokenCompany

client = TheTokenCompany(api_key="your-api-key")

result = client.compress(
    "This is a very long piece of text that contains a lot of unnecessary "
    "filler words and redundant information. When working with large language "
    "models, you often want to compress this kind of verbose content to save "
    "on token usage and reduce costs.",
    model="bear-1.2",
)

print(f"Compressed: {result.output}")
print(f"Tokens:     {result.input_tokens} → {result.output_tokens} ({result.compression_ratio:.1f}x)")
