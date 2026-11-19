"""Capture backend-verified forum quadruples in a new, immutable run directory.

This is a controlled forum component study, not a full VWA evaluation. Split
by forum before capturing; preserve every forum's initial subscription state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.parse import urljoin, urlparse


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def split_forums(forums, seed, test_forums=25, val_forums=10):
    names = sorted(set(forums))
    if len(names) < test_forums + val_forums + 10:
        raise ValueError("Need at least 10 training forums beyond val/test")
    random.Random(seed).shuffle(names)
    return {"test": names[:test_forums],
            "val": names[test_forums:test_forums + val_forums],
            "train": names[test_forums + val_forums:]}


def subscription(page):
    return page.evaluate("""() => {
      const f = [...document.forms].find(f => /\\/subscribe|\\/unsubscribe/.test(f.action));
      if (!f) throw Error('subscription form missing');
      const b = f.querySelector('button'); if (!b) throw Error('button missing');
      const r = b.getBoundingClientRect();
      return {subscribed: /\\/unsubscribe/.test(f.action),
              bbox: [r.x,r.y,r.right,r.bottom], text: b.innerText};
    }""")


def set_subscription(page, desired):
    before = subscription(page)
    if before["subscribed"] != desired:
        with page.expect_response(lambda r: r.request.method == "POST" and "subscribe" in r.url,
                                  timeout=15000) as response:
            page.locator('form.subscribe-form button').first.click()
        if response.value.status >= 400:
            raise RuntimeError("subscription POST failed")
        # Postmill's AJAX response can leave the old button/form in the DOM.
        # Confirm persisted backend state on a new page before assigning labels.
        page.reload(wait_until="domcontentloaded")
    after = subscription(page)
    if after["subscribed"] != desired:
        raise RuntimeError("backend subscription transition failed")
    return after


def capture(page, path, category, dx=0, dy=0):
    before = subscription(page)
    if category == "unrelated":
        page.evaluate("""() => {
          document.documentElement.style.backgroundColor = '#dce8f1';
          document.body.style.backgroundColor = '#dce8f1';
          document.querySelectorAll('img').forEach(x => x.style.filter='hue-rotate(113deg)');
        }""")
    elif category == "transform":
        page.evaluate("""([dx,dy]) => {
          const f=[...document.forms].find(f=>/\\/subscribe|\\/unsubscribe/.test(f.action));
          f.style.transform=`translate(${dx}px,${dy}px)`;
        }""", [dx, dy])
    after = subscription(page)
    if before["subscribed"] != after["subscribed"]:
        raise RuntimeError("non-target edit changed subscription")
    expected = [before["bbox"][i] + ([dx, dy, dx, dy][i]
                if category == "transform" else 0) for i in range(4)]
    if max(abs(a-b) for a, b in zip(after["bbox"], expected)) > 1:
        raise RuntimeError("layout mapping did not match the frozen translation")
    x0, y0, x1, y1 = after["bbox"]
    if not (0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720):
        raise RuntimeError("critical control is outside the viewport")
    visible = page.evaluate("""() => {
      const b=[...document.forms].find(f=>/\\/subscribe|\\/unsubscribe/.test(f.action)).querySelector('button');
      const r=b.getBoundingClientRect(),e=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
      return !!e && (e===b || b.contains(e));
    }""")
    if not visible:
        raise RuntimeError("edited control is occluded")
    page.screenshot(path=str(path))
    return {"image": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bbox": after["bbox"], "subscribed": after["subscribed"],
            "oracle": {"render": True, "state_preserved": True,
                       "control_executable": visible, "category": category}}


def discover(page, base, limit):
    queue, seen, forums = [base + "/forums"], set(), set()
    while queue and len(forums) < limit and len(seen) < 100:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        links = page.eval_on_selector_all("a[href]", "xs=>xs.map(x=>({href:x.href,text:x.innerText,rel:x.rel}))")
        for link in links:
            u = urlparse(link["href"])
            if u.netloc != urlparse(base).netloc:
                continue
            if u.path.startswith("/f/") and len(u.path.rstrip("/").split("/")) == 3:
                forums.add(u.path.rstrip("/"))
            if u.path.startswith("/forums") and link["href"] not in seen:
                queue.append(link["href"])
    return sorted(forums)[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--auth", required=True)
    ap.add_argument("--browser", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:9999")
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--forums", type=int, default=75)
    ap.add_argument("--test-forums", type=int, default=25)
    ap.add_argument("--val-forums", type=int, default=10)
    ap.add_argument("--split-manifest")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out / "dataset.jsonl").exists():
        raise RuntimeError("Use a new directory; do not replace a dataset")
    from playwright.sync_api import sync_playwright
    status = {"phase": "collecting", "captured_forums": 0, "failed_forums": [],
              "restoration_failures": [], "seed": args.seed}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, executable_path=args.browser)
        ctx = browser.new_context(storage_state=args.auth, viewport={"width": 1280, "height": 720})
        page = ctx.new_page()
        if args.split_manifest:
            frozen=json.loads(Path(args.split_manifest).read_text())
            if frozen["seed"] != args.seed: raise ValueError("Frozen seed changed")
            splits=frozen["forums"]
        else:
            forums = discover(page, args.base, args.forums)
            splits = split_forums(forums, args.seed, args.test_forums, args.val_forums)
        write_json(out / "split_manifest.json", {"seed": args.seed, "forums": splits,
                   "split_unit": "forum; all goals/states/edits remain together",
                   "test_cases_planned": 4 * len(splits["test"]),
                   "frozen_before_capture_or_model_scoring": True})
        with (out / "dataset.jsonl").open("x") as stream:
            for split, names in splits.items():
                for forum in names:
                    initial = None
                    try:
                        page.goto(urljoin(args.base, forum), wait_until="domcontentloaded", timeout=30000)
                        initial = subscription(page)["subscribed"]
                        states_file=out/"initial_states.json"
                        states=json.loads(states_file.read_text()) if states_file.exists() else {}
                        states[forum]=initial;write_json(states_file,states)
                        folder = out / "images" / forum.split("/")[-1]; folder.mkdir(parents=True)
                        frames = {}
                        for state in [False, True]:
                            set_subscription(page, state)
                            frames[state] = {}
                            for category in ["original", "unrelated", "transform"]:
                                page.reload(wait_until="domcontentloaded")
                                if subscription(page)["subscribed"] != state:
                                    raise RuntimeError("backend state did not survive reload")
                                frames[state][category] = capture(page, folder / f"{state}_{category}.png",
                                                                 category, dx=-48, dy=24)
                        for desired in [True, False]:
                            for alias in range(2):
                                goal = (("Subscribe to this forum.", "Make sure you are subscribed to this forum.")
                                        if desired else ("Unsubscribe from this forum.", "Make sure you are not subscribed to this forum."))[alias]
                                original, target = frames[not desired], frames[desired]
                                row = {"id": f"{forum.rsplit('/',1)[-1]}_{desired}_{alias}", "forum": forum,
                                       "split": split, "goal": goal, "size": [1280,720],
                                       "variants": {"original": dict(original["original"], action=0),
                                                    "target": dict(target["original"], action=1),
                                                    "unrelated": dict(original["unrelated"], action=0),
                                                    "transform": dict(original["transform"], action=0)},
                                       "tau": {"dx": -48/1280, "dy": 24/720},
                                       "target_backend_changed": True,
                                       "edit_pool": ["subscription_toggle", "background_hue", "control_translation"]}
                                for frame in row["variants"].values():
                                    frame["image"] = str(Path(frame["image"]).relative_to(out))
                                stream.write(json.dumps(row) + "\n"); stream.flush()
                        status["captured_forums"] += 1
                    except Exception as error:
                        status["failed_forums"].append({"forum": forum, "split": split, "error": str(error)})
                    finally:
                        if initial is not None:
                            try:
                                page.goto(urljoin(args.base, forum), wait_until="domcontentloaded")
                                set_subscription(page, initial)
                            except Exception as error:
                                status["restoration_failures"].append({"forum": forum, "error": str(error)})
                        write_json(out / "collection_status.json", status)
                        if status["restoration_failures"]:
                            raise RuntimeError("State restoration failed; stop further mutations")
        browser.close()
    rows = [json.loads(x) for x in (out / "dataset.jsonl").read_text().splitlines()]
    counts = {s: sum(r["split"] == s for r in rows) for s in splits}
    status.update(phase="finished" if not status["failed_forums"] else "partial; preserve planned denominator", counts=counts)
    write_json(out / "collection_status.json", status)
    # Never silently replace failed held-out forums or shrink the planned test set.
    write_json(out / "dataset_complete.json", {"sha256": hashlib.sha256((out/"dataset.jsonl").read_bytes()).hexdigest(),
               "counts": counts, "test_planned": len(splits["test"])*4,
               "eligible_for_training": all(counts.values()) and counts["test"] == len(splits["test"])*4 and not status["restoration_failures"],
               "scope": "live-rendered, backend-verified forum component subset; not VWA end-to-end"})


if __name__ == "__main__":
    main()
