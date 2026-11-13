"""Score the base model's point predictions on the official annotated screenshots."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from PIL import Image
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

root = Path(sys.argv[1])
annotations = [row for file in sorted((root / "dataset/annotations").glob("*.json"))
               for row in json.loads(file.read_text())]
if not annotations:
    raise SystemExit("No ScreenSpot-Pro annotations found")

def area_bucket(area):
    if area < 0.001:
        return "<0.1%"
    if area < 0.01:
        return "0.1-1%"
    if area <= 0.05:
        return "1-5%"
    return ">5%"

def predict(row):
    x0, y0, x1, y1 = row["bbox"]
    width, height = row["img_size"]
    record = {"id": row["id"], "bucket": area_bucket((x1-x0)*(y1-y0)/(width*height))}
    file = root / "dataset/images" / row["img_filename"]
    with Image.open(file) as img:
        if img.size != (width, height):
            raise ValueError(f"Annotated image size differs for {row['id']}")
        buffer = BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=93)
    uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    instruction = (
        "Locate the interface element for the given instruction in the image. "
        "Return the location as JSON with x and y, both normalized to [0,1000] "
        "from the top-left corner. Return only a JSON object. "
        "The task instruction is: " + row["instruction"]
    )
    payload = {"model": os.environ["CAUSALACT_BASE_MODEL"], "messages": [{"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": uri}},
    ]}]}
    request = urllib.request.Request(
        os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=100) as response:
                result = json.load(response)
            content = result["choices"][0]["message"]["content"] or ""
            match = re.search(r'\{[^{}]*\}', content)
            if not match:
                return dict(record, status="invalid_prediction", hit=False)
            try:
                point = json.loads(match[0])
                x, y = float(point["x"]), float(point["y"])
            except (ValueError, TypeError, KeyError):
                return dict(record, status="invalid_prediction", hit=False)
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1000 and 0 <= y <= 1000):
                return dict(record, status="invalid_prediction", hit=False)
            px, py = x * width / 1000, y * height / 1000
            return dict(record, status="scored", point=[px, py],
                        hit=x0 <= px <= x1 and y0 <= py <= y1)
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}"
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (OSError, TimeoutError, KeyError, ValueError, IndexError) as exc:
            error = type(exc).__name__
        time.sleep(2 * (attempt + 1))
    return dict(record, status="api_error", error=error)

out = root / "point_results.jsonl"
done = set()
if out.exists():
    for line in out.read_text().splitlines():
        previous = json.loads(line)
        if previous["status"] != "api_error":
            done.add(previous["id"])
pending = [row for row in annotations if row["id"] not in done]
with out.open("a") as f, ThreadPoolExecutor(max_workers=4) as pool:
    first = pending[:4]
    first_results = list(pool.map(predict, first))
    for result in first_results:
        f.write(json.dumps(result) + "\n")
    f.flush()
    if first_results and all(result["status"] == "api_error" for result in first_results):
        raise SystemExit(f"All preflight requests failed: {first_results[0]['error']}")
    for index, result in enumerate(pool.map(predict, pending[len(first):]), len(first) + 1):
        f.write(json.dumps(result) + "\n")
        f.flush()
        if index % 25 == 0:
            print(f"New predictions {index}/{len(pending)}", flush=True)
results = {}
for line in out.read_text().splitlines():
    row = json.loads(line)
    results[row["id"]] = row
summary = {}
for name in ("<0.1%", "0.1-1%", "1-5%", ">5%"):
    bucket = [row for row in results.values() if row["bucket"] == name]
    scored = [row for row in bucket if row["status"] != "api_error"]
    hits = sum(row["hit"] for row in scored)
    summary[name] = {"total": len(bucket), "scored": len(scored), "hits": hits,
                     "point_accuracy": hits/len(scored) if scored else None}
(root/"point_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(summary)
