"""Apply the relay's existing image encoding before its ingress size check."""

import base64
from copy import deepcopy
import io

from PIL import Image


IMAGE_BYTE_LIMIT = 3_700_000


def normalize_image(mime, encoded, limit=IMAGE_BYTE_LIMIT):
    """Match the relay's JPEG policy exactly; retain the original dimensions."""
    if len(encoded) <= (limit // 3) * 4:
        return mime, encoded
    original = base64.b64decode(encoded, validate=True)
    if len(original) <= limit:
        return mime, encoded
    with Image.open(io.BytesIO(original)) as source:
        image = source.convert("RGB")
        for quality in (94, 90, 85, 75, 60, 45):
            result = io.BytesIO()
            image.save(result, "JPEG", quality=quality, optimize=True)
            if result.tell() <= limit:
                return "image/jpeg", base64.b64encode(result.getvalue()).decode()
    raise ValueError("Image exceeds the Bedrock byte limit")


def normalize_messages(messages, limit=IMAGE_BYTE_LIMIT):
    """Return a copy and conversion metadata without retaining image content."""
    result = deepcopy(messages)
    audit = []
    for message_index, message in enumerate(result):
        if not isinstance(message.get("content"), list):
            continue
        for block_index, block in enumerate(message["content"]):
            if block.get("type") != "image_url":
                continue
            url = block["image_url"]["url"]
            if not url.startswith("data:image/") or ";base64," not in url:
                continue
            prefix, encoded = url.split(",", 1)
            mime = prefix[5:].split(";", 1)[0]
            new_mime, new_encoded = normalize_image(mime, encoded, limit)
            if new_encoded == encoded:
                continue
            original = base64.b64decode(encoded, validate=True)
            normalized = base64.b64decode(new_encoded, validate=True)
            with Image.open(io.BytesIO(original)) as source:
                dimensions = source.size
            with Image.open(io.BytesIO(normalized)) as source:
                if source.size != dimensions:
                    raise ValueError("Image normalization changed pixel dimensions")
            block["image_url"]["url"] = "data:" + new_mime + ";base64," + new_encoded
            audit.append({
                "message_index": message_index,
                "block_index": block_index,
                "original_bytes": len(original),
                "sent_bytes": len(normalized),
                "width": dimensions[0],
                "height": dimensions[1],
                "encoding": new_mime,
            })
    return result, audit
