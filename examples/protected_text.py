"""Protect structural markers from compression using ttc_safe tags."""

from thetokencompany import TheTokenCompany, protect

client = TheTokenCompany(api_key="your-api-key")

conversation = (
    f"{protect('system:')} You are a helpful assistant that answers questions concisely.\n"
    f"{protect('user:')} Can you explain the theory of relativity in simple terms?\n"
    f"{protect('assistant:')} "
)

result = client.compress(conversation, model="bear-1.2", aggressiveness=0.7)
print(result.output)
