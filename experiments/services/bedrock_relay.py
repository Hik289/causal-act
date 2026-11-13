"""Local Chat Completions adapter for the Bedrock Messages endpoint."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hmac
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request
import uuid

MODEL = os.environ["CAUSALACT_BEDROCK_MODEL"]
PROFILE = os.environ.get("CAUSALACT_BEDROCK_PROFILE", "global." + MODEL)
REGION = os.environ.get("AWS_REGION", "us-east-1")
URL = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{PROFILE}/invoke"
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
aws_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
if not aws_key:
    raise SystemExit("No Bedrock credential supplied")
client_key = os.environ["OPENAI_API_KEY"].strip()
if not client_key:
    raise SystemExit("Set a non-empty local client credential")
local_auth = "Bearer " + client_key
slots = threading.BoundedSemaphore(8)
audit_lock = threading.Lock()
request_path = re.compile(r"/(?:worker/([a-z0-9_]+)/)?v1/chat/completions")
image_data = re.compile(r"data:(image/(?:jpeg|png|gif|webp));base64,(.+)", re.DOTALL)

def request_body(payload):
    system, turns = [], []
    for message in payload["messages"]:
        role = message["role"]
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        blocks = []
        for item in content or []:
            if item["type"] == "text":
                if item["text"]:
                    blocks.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                match = image_data.fullmatch(item["image_url"]["url"])
                if not match:
                    raise ValueError("Screenshots must be inline supported images")
                mime, data = match.groups()
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": mime, "data": data}})
            else:
                raise ValueError("Unsupported content type")
        if role in ("system", "developer"):
            if any(block["type"] != "text" for block in blocks):
                raise ValueError("System messages must be text")
            system.extend(blocks)
        elif role in ("user", "assistant"):
            if blocks:
                turns.append({"role": role, "content": blocks})
        else:
            raise ValueError("Unsupported message role")
    if not turns or turns[0]["role"] != "user":
        raise ValueError("A user message must come first")
    body = {"anthropic_version": "bedrock-2023-05-31", "messages": turns,
            "max_tokens": int(payload.get("max_completion_tokens") or
                              payload.get("max_tokens") or 2048)}
    if system:
        body["system"] = system
    return body

def audit(worker, status, usage=None, error=None):
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "worker": worker, "model": MODEL, "status": status}
    if usage:
        record.update(input_tokens=usage.get("input_tokens"),
                      output_tokens=usage.get("output_tokens"))
    if error:
        record["error_type"] = error
    with audit_lock, (root / "requests.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def error(self, status, code):
        self.send_json(status, {"error": {"message": code, "type": code, "code": code}})

    def do_POST(self):
        match = request_path.fullmatch(self.path)
        if not match:
            self.error(404, "unknown_route")
            return
        if not hmac.compare_digest(self.headers.get("Authorization", ""), local_auth):
            self.error(401, "invalid_local_credential")
            return
        worker = match[1] or "probe"
        try:
            length = int(self.headers["Content-Length"])
            if not 0 < length <= 28 * 1024 * 1024:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            body = request_body(payload)
            if payload.get("stream"):
                raise ValueError("Streaming is disabled")
        except (TypeError, AttributeError, ValueError, KeyError):
            audit(worker, 400, error="invalid_request")
            self.error(400, "invalid_request")
            return
        request = urllib.request.Request(URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + aws_key})
        if not slots.acquire(timeout=95):
            self.error(504, "local_queue_timeout")
            return
        try:
            with urllib.request.urlopen(request, timeout=105) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            kind = exc.headers.get("x-amzn-errortype", "bedrock_error").split(":")[0]
            detail = exc.read(16384)
            if exc.code == 400 and re.search(rb"context length|too many tokens|input is too long", detail, re.I):
                kind = "context_length_exceeded"
            audit(worker, exc.code, error=kind)
            self.error(exc.code, kind)
            return
        except (OSError, ValueError):
            audit(worker, 504, error="bedrock_connection_error")
            self.error(504, "bedrock_connection_error")
            return
        finally:
            slots.release()

        output = "".join(block["text"] for block in result.get("content", [])
                         if block.get("type") == "text")
        usage = result.get("usage", {})
        audit(worker, 200 if output else 502, usage, error=None if output else "empty_completion")
        if not output:
            self.error(502, "empty_completion")
            return
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        self.send_json(200, {"id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion", "created": int(datetime.now(timezone.utc).timestamp()),
            "model": MODEL, "choices": [{"index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "length" if result.get("stop_reason") == "max_tokens" else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}})

ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("CAUSALACT_BEDROCK_PORT", "18822"))), Handler).serve_forever()
