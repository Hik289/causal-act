"""AndroidWorld inference interface backed by an existing shared paced relay.

This module starts neither a relay nor an emulator. A caller must supply the
already admitted Claude endpoint and classify InvalidModelCall as an invalid
attempt, including when AndroidWorld is generating an action summary.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

MODEL = "anthropic.claude-opus-5-5"
RETRYABLE = frozenset({408, 429, 500, 502, 503, 504, 529})
REQUEST_LIMIT = 28 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


class InvalidModelCall(RuntimeError):
    """A transport/provider failure, never a scored model task failure."""

    def __init__(self, code, call_id, attempts):
        self.code, self.call_id, self.attempts = code, call_id, attempts
        super().__init__(f"Claude inference invalid: {code}; call {call_id}; attempts {attempts}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PacedClaudeWrapper:
    """Duck-typed native LlmWrapper/MultimodalLlmWrapper implementation.

    encode_image must be the pinned AndroidWorld array_to_jpeg_bytes function.
    The native factory below supplies it. The shared relay owns concurrency and
    request pacing; this client does not create additional inference slots.
    Retries are explicit and bounded, with no hidden fallback to another model.
    """

    def __init__(self, *, endpoint, credential, audit_dir, encode_image,
                 timeout=230, max_attempts=1, retry_delays=(), sleep=time.sleep):
        url = urllib.parse.urlsplit(endpoint)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1"}
                or not url.port or url.username or url.password or url.query or url.fragment
                or not re.fullmatch(r"/worker/android_[a-z0-9_]+/v1/chat/completions", url.path)):
            raise ValueError("Use an explicit loopback Android worker route on the shared paced relay")
        if not isinstance(credential, str) or not credential.strip() or any(c in credential for c in "\r\n"):
            raise ValueError("A non-empty local relay credential is required")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be explicitly bounded from 1 to 3")
        if len(retry_delays) != max_attempts - 1 or any(not math.isfinite(d) or d < 0 for d in retry_delays):
            raise ValueError("Supply one nonnegative delay for each permitted retry")
        if not 0 < timeout <= 300 or not callable(encode_image):
            raise ValueError("A bounded timeout and native image encoder are required")
        self.endpoint, self._credential = endpoint, credential.strip()
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.encode_image = encode_image
        self.timeout, self.max_attempts = timeout, max_attempts
        self.retry_delays, self.sleep = tuple(retry_delays), sleep
        self.model = MODEL
        self._audit_lock = threading.Lock()
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _write(self, path, data):
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(0o600)

    def _event(self, value):
        value = dict(value, recorded_utc=datetime.now(timezone.utc).isoformat(), score=None)
        with self._audit_lock, (self.audit_dir / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def predict(self, text_prompt):
        return self.predict_mm(text_prompt, [])

    def predict_mm(self, text_prompt, images):
        call_id = uuid.uuid4().hex
        folder = self.audit_dir / call_id
        folder.mkdir(mode=0o700)
        if not isinstance(text_prompt, str) or not text_prompt:
            self._event(dict(call_id=call_id, valid=False, code="invalid_prompt", attempts=0))
            raise InvalidModelCall("invalid_prompt", call_id, 0)
        content = [{"type": "text", "text": text_prompt}]
        image_receipts = []
        try:
            for image in images:
                encoded = self.encode_image(image)
                if not isinstance(encoded, bytes) or not encoded.startswith(b"\xff\xd8") or not encoded.endswith(b"\xff\xd9"):
                    raise ValueError("The native encoder must return JPEG bytes")
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")}})
                source = image.tobytes() if hasattr(image, "tobytes") else None
                image_receipts.append(dict(encoded_sha256=digest(encoded), encoded_bytes=len(encoded),
                    array_sha256=digest(source) if source is not None else None,
                    shape=list(image.shape) if hasattr(image, "shape") else None))
        except Exception as error:
            self._event(dict(call_id=call_id, valid=False, code="invalid_native_image", attempts=0))
            raise InvalidModelCall("invalid_native_image", call_id, 0) from error
        payload = dict(model=MODEL, messages=[dict(role="user", content=content)], max_tokens=1000)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > REQUEST_LIMIT:
            self._event(dict(call_id=call_id, valid=False, code="request_too_large", attempts=0))
            raise InvalidModelCall("request_too_large", call_id, 0)
        self._write(folder / "request.json", body)
        receipt = dict(call_id=call_id, endpoint=self.endpoint, model=MODEL,
            started_utc=datetime.now(timezone.utc).isoformat(),
            request_sha256=digest(body), prompt_sha256=digest(text_prompt.encode("utf-8")),
            images=image_receipts, requested_max_tokens=1000,
            sampling="shared relay defaults; no temperature or seed claim")
        self._write(folder / "input_receipt.json", json.dumps(receipt, indent=2).encode())
        for attempt in range(1, self.max_attempts + 1):
            status, raw, code, retryable, retry_after = None, b"", "connection_error", True, 0
            request = urllib.request.Request(self.endpoint, data=body, headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + self._credential})
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    status = response.status
                    raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    code, retryable = "response_too_large", False
                elif status != 200:
                    code, retryable = "http_" + str(status), status in RETRYABLE
                else:
                    try:
                        result = json.loads(raw)
                        if result.get("model") != MODEL:
                            code, retryable = "model_mismatch", False
                        else:
                            text = result["choices"][0]["message"]["content"]
                            if not isinstance(text, str) or not text.strip():
                                code, retryable = "empty_completion", True
                            else:
                                self._write(folder / (f"response_attempt{attempt}.json"), raw)
                                self._event(dict(receipt, valid=True, attempt=attempt, http_status=status,
                                    response_sha256=digest(raw), usage=result.get("usage"),
                                    finish_reason=result["choices"][0].get("finish_reason")))
                                # Action syntax, infeasibility and nonempty truncated output
                                # stay with the unmodified native parser and scorer.
                                return text, None, result
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                        code, retryable = "invalid_response_schema", False
            except urllib.error.HTTPError as error:
                status = error.code
                raw = error.read(16384)
                code, retryable = "http_" + str(status), status in RETRYABLE
                try:
                    retry_after = float(error.headers.get("Retry-After", 0))
                    if not math.isfinite(retry_after) or retry_after < 0:
                        retry_after = 0
                except (ValueError, TypeError):
                    retry_after = 0
            except (OSError, urllib.error.URLError, TimeoutError, http.client.HTTPException):
                code, retryable = "connection_error", True
            self._write(folder / (f"invalid_attempt{attempt}.response.bin"), raw)
            self._write(folder / (f"invalid_attempt{attempt}.json"), json.dumps(dict(
                valid=False, score=None, code=code, http_status=status,
                response_sha256=digest(raw), response_bytes=len(raw)), indent=2).encode())
            self._event(dict(receipt, valid=False, code=code, attempt=attempt, http_status=status))
            if not retryable or attempt == self.max_attempts:
                raise InvalidModelCall(code, call_id, attempt)
            self.sleep(max(retry_after, self.retry_delays[attempt - 1]))


def make_native_wrapper(**settings):
    """Bind the pinned AndroidWorld interfaces and its exact JPEG encoder.

    The caller owns AndroidWorld source/import pinning. This function does not
    patch native prompts, parsing, action execution, timing or task scoring.
    """
    from android_world.agents import infer

    class NativeClaudeWrapper(PacedClaudeWrapper, infer.LlmWrapper, infer.MultimodalLlmWrapper):
        pass

    return NativeClaudeWrapper(encode_image=infer.array_to_jpeg_bytes, **settings)
