from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from random import Random
import json
import sys
import xml.etree.ElementTree as ET

data = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
seed = 20261112
rng = Random(seed)

def read_routes(name):
    return ET.parse(data / name).getroot().findall("route")

def scene(route):
    scenes = route.findall("./scenarios/scenario")
    assert len(scenes) == 1
    return scenes[0].get("type")

def triggers(route):
    for node in route.findall("./scenarios/scenario"):
        loc = node.find("trigger_point")
        if loc is not None:
            yield (route.get("town"), node.get("type"), *(round(float(loc.get(k)),1) for k in ("x","y","z")))

legacy_triggers = {t for route in read_routes("routes_validation.xml") for t in triggers(route)}
dev_ids = {r.get("id") for r in read_routes("drivetransformer_bench2drive_dev10.xml")}
source = data / "bench2drive_0.0.4_val.xml"
pool = read_routes(source.name)
assert len(pool) == len({r.get("id") for r in pool}) == 220
groups = defaultdict(list)
excluded = []
for route in pool:
    if route.get("id") in dev_ids or any(t in legacy_triggers for t in triggers(route)):
        excluded.append(route.get("id"))
        continue
    groups[scene(route)].append(route)
assert sum(map(len, groups.values())) >= 120
types = sorted(groups)
for name in types:
    rng.shuffle(groups[name])
counts = {name: min(2, len(groups[name])) for name in types}
while sum(counts.values()) < 100:
    eligible = [name for name in types if counts[name] < len(groups[name])]
    rng.shuffle(eligible)
    for name in eligible[:100-sum(counts.values())]:
        counts[name] += 1
holdout = [r for name in types for r in groups[name][:counts[name]]]
holdout_ids = {r.get("id") for r in holdout}
available = {name: [r for r in groups[name] if r.get("id") not in holdout_ids] for name in types}
tune_types = rng.sample([name for name in types if available[name]], 20)
validation = [available[name][0] for name in tune_types]
assert len(holdout) == 100 and len(validation) == 20
assert not ({r.get("id") for r in validation} & holdout_ids)
assert not ({t for r in validation for t in triggers(r)} & {t for r in holdout for t in triggers(r)})

manifest = {"benchmark":"Bench2Drive 0.0.4", "sampling_seed": seed,
            "traffic_manager_seed": 0, "source": source.name,
            "source_sha256": sha256(source.read_bytes()).hexdigest(),
            "excluded_dev10_or_legacy_trigger_ids": sorted(excluded),
            "created_utc": datetime.now(timezone.utc).isoformat(), "splits": {}}
for name, selected in (("validation20", validation), ("holdout100", holdout)):
    ids = {r.get("id") for r in selected}
    ordered = [r for r in pool if r.get("id") in ids]
    root = ET.Element("routes")
    root.extend(ordered)
    path = out / (name + ".xml")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    manifest["splits"][name] = {
        "route_ids": [r.get("id") for r in ordered],
        "scenario_counts": dict(sorted(Counter(map(scene, ordered)).items())),
        "xml_sha256": sha256(path.read_bytes()).hexdigest(),
    }
(out / "split_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("Excluded",len(excluded),"routes with Dev10 IDs or legacy trigger points")
print("Validation",len(validation),"routes and",len(tune_types),"types")
print("Holdout",len(holdout),"routes and",len(groups),"types")
