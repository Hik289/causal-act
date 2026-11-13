"""Select the recorded forum tasks from a locally configured VisualWebArena task file."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("task_file", type=Path, help="JSON list of locally configured forum tasks")
parser.add_argument("output", type=Path)
args = parser.parse_args()

tasks = {task["task_id"]: task for task in json.loads(args.task_file.read_text())}
ids = json.loads(Path(__file__).with_name("forum_task_ids.json").read_text())
missing = set(ids) - tasks.keys()
if missing:
    parser.error(f"{len(missing)} recorded task IDs are missing from the input file")
args.output.mkdir(parents=True, exist_ok=True)
files = {args.output / f"{index}.json": json.dumps(tasks[task_id], indent=2) + "\n"
         for index, task_id in enumerate(ids)}
if any(path.exists() and path.read_text() != content for path, content in files.items()):
    parser.error("The output directory contains a different task configuration")
for path, content in files.items():
    path.write_text(content)
print(f"Prepared {len(files)} forum tasks")
