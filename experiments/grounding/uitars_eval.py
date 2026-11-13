"""Run the existing local UI-TARS model on official ScreenSpot-Pro images."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import os
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from PIL import Image
from transformers import AutoProcessor

processor, dataset, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
image_processor = AutoProcessor.from_pretrained(processor).image_processor
prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""
rows = [row for p in sorted((dataset / "annotations").glob("*.json"))
        for row in json.loads(p.read_text())]
assert len(rows) == 1581

def score(row):
    width, height = row["img_size"]
    x0, y0, x1, y1 = row["bbox"]
    area = (x1-x0) * (y1-y0) / (width*height)
    bucket = "<0.1%" if area < .001 else "0.1-1%" if area < .01 else "1-5%" if area <= .05 else ">5%"
    record = {"id": row["id"], "bucket": bucket}
    with Image.open(dataset / "images" / row["img_filename"]) as image:
        image = image.convert("RGB")
        image.thumbnail((640, 640))
        grid = image_processor(images=image, return_tensors="np")["image_grid_thw"][0]
        input_height = int(grid[1]) * image_processor.patch_size
        input_width = int(grid[2]) * image_processor.patch_size
        data = BytesIO()
        image.save(data, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    payload = {"model": "ui-tars-1.5-7b", "max_tokens": 64,
               "messages": [{"role": "user", "content": [
                   {"type": "text", "text": prompt + row["instruction"]},
                   {"type": "image_url", "image_url": {"url": uri}},
               ]}]}
    req = urllib.request.Request(os.environ["CAUSALACT_GROUNDER_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("CAUSALACT_GROUNDER_KEY", "unused")})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=100) as response:
                result = json.load(response)["choices"][0]["message"]["content"]
            point = re.search(r"<point>\s*(\d+)\s*[, ]\s*(\d+)\s*</point>", result)
            if point is None:
                point = re.search(r"(?:start_box|start_point|point)\s*=\s*['\"]?\(\s*(\d+)\s*,\s*(\d+)\s*\)", result)
            if point is None:
                return dict(record, status="invalid_prediction", hit=False)
            px, py = int(point[1]), int(point[2])
            if not (0 <= px < input_width and 0 <= py < input_height):
                return dict(record, status="invalid_prediction", hit=False)
            x, y = px * width / input_width, py * height / input_height
            return dict(record, status="scored", hit=x0 <= x <= x1 and y0 <= y <= y1,
                        prediction_px=[px,py], image_grid_px=[input_width,input_height])
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            error = f"HTTP {exc.code}" if isinstance(exc, urllib.error.HTTPError) else type(exc).__name__
            time.sleep(2**attempt)
    return dict(record, status="api_error", error=error)

def worker(index):
    path = out / f"worker_{index}.jsonl"
    old = {row["id"]: row for row in map(json.loads, path.read_text().splitlines())
           if row["status"] != "api_error"} if path.exists() else {}
    shard = rows[index::2]
    consecutive = 0
    with path.open("a") as stream:
        for row in shard:
            if row["id"] in old:
                continue
            result = score(row)
            stream.write(json.dumps(result)+"\n")
            stream.flush()
            if result["status"] == "api_error":
                consecutive += 1
                if consecutive >= 3:
                    raise RuntimeError(f"Worker {index} has three consecutive server errors")
            else:
                consecutive = 0
                old[row["id"]] = result
                if len(old) % 25 == 0:
                    print(f"Worker {index}: {len(old)}/{len(shard)}", flush=True)

checks = [score(rows[0]), score(rows[-1])]
print("Local smoke:",[r["status"] for r in checks],flush=True)
if not any(r["status"] == "scored" for r in checks):
    raise SystemExit("No valid point in UI-TARS smoke")
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(worker, range(2)))
results = {}
for path in out.glob("worker_*.jsonl"):
    for row in map(json.loads, path.read_text().splitlines()):
        results[row["id"]] = row
counts = Counter(r["bucket"] for r in results.values())
summary = {}
for bucket, total in counts.items():
    scored = [r for r in results.values() if r["bucket"] == bucket and r["status"] != "api_error"]
    summary[bucket] = {"total":total, "scored":len(scored), "hits":sum(r["hit"] for r in scored)}
(out / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
