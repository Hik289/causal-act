"""Optional private upstream diagnostics; never changes inference or responses."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


def record_upstream(*, worker, model, request_bytes, response,
                    output_dir=None, workers=None):
    """Record only explicitly selected diagnostic workers, with no request body.

    The provider response is private evidence and can contain generated text.
    Authorization headers and credentials are not inputs to this function.
    Diagnostics failures must not cause inference failures or extra requests.
    """
    if workers is None:
        workers = {v for v in os.environ.get("CAUSALACT_DIAGNOSTIC_WORKERS", "").split(",") if v}
    if output_dir is None:
        output_dir = os.environ.get("CAUSALACT_DIAGNOSTIC_DIR")
    if worker not in workers or not output_dir:
        return None
    try:
        encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), worker=worker,
            model=model, request_sha256=hashlib.sha256(request_bytes).hexdigest(),
            response_sha256=hashlib.sha256(encoded).hexdigest(), provider_response=response,
            scope="private provider diagnostics, not benchmark scoring")
        directory = Path(output_dir)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / (uuid.uuid4().hex + ".json")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return str(path)
    except (OSError, TypeError, ValueError):
        return None
