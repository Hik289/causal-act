import base64
from copy import deepcopy
import io
import random
import unittest

from PIL import Image

from bedrock_image_transport import normalize_image, normalize_messages


def image_url(image):
    stream = io.BytesIO()
    image.save(stream, "PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


class ImageTransportTests(unittest.TestCase):
    def test_small_images_and_other_message_fields_are_unchanged(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Inspect this image."},
            {"type": "image_url", "image_url": {
                "url": image_url(Image.new("RGB", (8, 8))), "detail": "high"
            }},
        ]}]
        normalized, audit = normalize_messages(messages)
        self.assertEqual(normalized, messages)
        self.assertEqual(audit, [])

    def test_oversized_image_keeps_dimensions_and_does_not_mutate_input(self):
        random_bytes = random.Random(0).randbytes(128 * 128 * 3)
        image = Image.frombytes("RGB", (128, 128), random_bytes)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url(image)}}
        ]}]
        before = deepcopy(messages)
        normalized, audit = normalize_messages(messages, limit=40_000)
        self.assertEqual(messages, before)
        self.assertEqual(len(audit), 1)
        self.assertEqual((audit[0]["width"], audit[0]["height"]), (128, 128))
        self.assertLessEqual(audit[0]["sent_bytes"], 40_000)
        self.assertTrue(normalized[0]["content"][0]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        ))

    def test_impossible_limit_raises_instead_of_truncating_image_data(self):
        encoded = image_url(Image.new("RGB", (8, 8))).split(",", 1)[1]
        with self.assertRaises(ValueError):
            normalize_image("image/png", encoded, limit=1)


if __name__ == "__main__":
    unittest.main()
