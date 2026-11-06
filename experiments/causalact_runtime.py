"""Local paths for experiment code. The launcher exports matching shell variables."""
from pathlib import Path
import os

ROOT = Path(os.environ.get("CAUSALACT_ROOT", Path(__file__).resolve().parents[1]))
RUN = Path(os.environ.get("CAUSALACT_RUN_DIR", ROOT / ".runs"))
VENDOR = ROOT / ".vendor"
VWA = Path(os.environ.get("CAUSALACT_VWA_DIR", VENDOR / "VisualWebArena"))

DEFAULTS = {
    "CAUSALACT_ROOT": ROOT,
    "CAUSALACT_RUN_DIR": RUN,
    "CAUSALACT_VWA_DIR": VWA,
    "CAUSALACT_VWA_AUTH": VWA / ".auth" / "reddit_state.json",
    "CAUSALACT_OSWORLD_DIR": VENDOR / "OSWorld",
    "CAUSALACT_BENCH2DRIVE_DIR": VENDOR / "Bench2Drive",
    "CAUSALACT_CARLA_DIR": VENDOR / "CARLA",
    "CAUSALACT_WEB_SCRIPTS": ROOT / "experiments" / "web",
    "CAUSALACT_WEB_AUX": ROOT / "experiments" / "web" / "operations",
}

def location(name, relative=""):
    return str(Path(os.environ.get(name, DEFAULTS[name])).expanduser() / relative)
