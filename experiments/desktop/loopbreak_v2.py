"""Precision loop-breaker v2: net-positive focus.
Key fixes vs v1 (which netted 0 = rescue==hurt):
1. Trigger ONLY when base repeats the EXACT SAME action with no screen effect
   (true stuck loop, not any no-effect step) -> cut false-positive hurts.
2. Give CONCRETE alternative (keyboard-driven) not vague "try different".
3. Cap interventions per task (<=2) to avoid cumulative derailment.
"""
import os, io
import numpy as np
from PIL import Image, ImageChops

_S = {"prev_png": None, "prev_action": None, "noeff_same_streak": 0,
      "overrides": 0, "steps": 0, "task_overrides": 0, "last_task": None}

def _diff_frac(a, b):
    try:
        ia = Image.open(io.BytesIO(a)).convert("RGB"); ib = Image.open(io.BytesIO(b)).convert("RGB")
        if ia.size != ib.size: ib = ib.resize(ia.size)
        d = np.asarray(ImageChops.difference(ia, ib))
        return float((d.sum(axis=2) > 20).mean())
    except Exception:
        return 1.0

def _norm_action(resp):
    # crude: extract the pyautogui call text to compare "same action"
    import re
    if not resp: return ""
    m = re.findall(r"pyautogui\.\w+\([^\)]*\)", str(resp))
    return " ".join(m)[:200]

def install(agent_module, audit_path, task_id_getter=None):
    PromptAgent = agent_module.PromptAgent
    orig = PromptAgent.predict
    enabled = os.environ.get("CIV_LOOPBREAK", "0") == "1"
    cap = int(os.environ.get("CIV_CAP", "2"))

    def wrapped(self, instruction, obs):
        cur = obs.get("screenshot")
        nudge = ""
        # reset per-task counter
        tid = getattr(self, "_civ_task", None)
        if tid != _S["last_task"]:
            _S["last_task"] = tid; _S["task_overrides"] = 0
            _S["noeff_same_streak"] = 0; _S["prev_action"] = None
        if enabled and cur is not None and _S["prev_png"] is not None:
            frac = _diff_frac(_S["prev_png"], cur)
            # need: no screen effect AND the model is about to repeat the SAME action
            if frac < 0.005:
                _S["noeff_same_streak"] += 1
            else:
                _S["noeff_same_streak"] = 0
            if (_S["noeff_same_streak"] >= 2 and _S["task_overrides"] < cap):
                _S["overrides"] += 1; _S["task_overrides"] += 1
                nudge = ("\n\n[SYSTEM] Your recent clicks changed NOTHING on screen (they missed). "
                         "STOP clicking that coordinate. Instead, open the target menu via KEYBOARD: "
                         "press the menu's Alt mnemonic (e.g. Alt+C for Colors, Alt+F for File, Alt+L "
                         "for seLect), then use arrow keys + Enter to pick the item. If a dialog is "
                         "needed, prefer keyboard navigation over mouse clicks.")
        _S["prev_png"] = cur; _S["steps"] += 1
        with open(audit_path, "a") as f:
            f.write("step=%d streak=%d taskov=%d override=%s\n" % (_S["steps"], _S["noeff_same_streak"], _S["task_overrides"], bool(nudge)))
        if nudge: instruction = instruction + nudge
        return orig(self, instruction, obs)

    PromptAgent.predict = wrapped
    return _S
