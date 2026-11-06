#!/usr/bin/env python3
"""Apply a benchmark patch to a clean checkout at its recorded revision."""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def git(target, *args):
    return subprocess.run(["git", "-C", str(target), *args], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark", choices=["visualwebarena", "osworld", "bench2drive"])
    p.add_argument("checkout", type=Path)
    args = p.parse_args()
    name, target = args.benchmark, args.checkout.resolve()
    repo = ROOT / "integrations"
    rev = (repo / f"{name}.revision").read_text().strip()
    patch = repo / f"{name}.patch"
    if git(target, "rev-parse", "HEAD") != rev:
        p.error(f"Expected {name} checkout at {rev}.")
    if git(target, "status", "--porcelain"):
        p.error("Use a clean checkout; this command writes benchmark integration files.")
    if name == "visualwebarena":
        source = ROOT / "experiments" / "web" / "integration"
        additions = [(source / "reddit_login.py", target / "reddit_login.py")]
    elif name == "osworld":
        source = ROOT / "experiments" / "desktop"
        additions = [(f, target / f.relative_to(source))
                     for f in source.rglob("*")
                     if f.is_file() and f.suffix in {".sh", ".py", ".json"}]
    else:
        source = ROOT / "experiments" / "driving"
        additions = [(source / "causalact_driving_agent.py",
                      target / "leaderboard/leaderboard/autoagents/causalact_driving_agent.py"),
                     (source / "routes/routes_devtest.xml", target / "leaderboard/data/routes_devtest.xml")]
    conflicts = [str(out.relative_to(target)) for f, out in additions
                 if out.exists() and f.read_bytes() != out.read_bytes()]
    if conflicts:
        p.error("Refusing to replace existing files: " + ", ".join(conflicts))
    git(target, "apply", "--check", str(patch))
    git(target, "apply", str(patch))
    for f, out in additions:
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    print(f"Applied {name} integration at {rev[:12]}.")

if __name__ == "__main__":
    main()
