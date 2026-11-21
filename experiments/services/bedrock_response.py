"""Classify a provider response without retrying or inventing agent actions."""


def completion_text(response):
    """A provider refusal must never be passed to an agent action parser."""
    if response.get("stop_reason") == "refusal":
        return "", "provider_refusal"
    output = "".join(block["text"] for block in response.get("content", [])
                     if block.get("type") == "text")
    return output, None if output else "empty_completion"
