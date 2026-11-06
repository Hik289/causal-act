"""Loop-breaker refiner for gimp (deterministic screen-diff based).
Monkeypatches PromptAgent.predict to detect no-effect looping and inject a
corrective nudge. Toggle via env CIV_LOOPBREAK=1.

Targets diagnosed gimp failure: base clicks, screen doesn't change (73% no-effect),
base repeats same action (loop). We detect: last screenshot ~= prev screenshot
(action had no effect) AND agent about to act again -> inject instruction telling
it the last action had no effect and to try a DIFFERENT method.
"""
import os, io, hashlib
import numpy as np
from PIL import Image, ImageChops

_STATE = {"prev_png": None, "noeffect_streak": 0, "overrides": 0, "steps": 0}

def _diff_frac(png_a, png_b):
    try:
        ia = Image.open(io.BytesIO(png_a)).convert("RGB")
        ib = Image.open(io.BytesIO(png_b)).convert("RGB")
        if ia.size != ib.size: ib = ib.resize(ia.size)
        d = np.asarray(ImageChops.difference(ia, ib))
        return float((d.sum(axis=2) > 20).mean())
    except Exception:
        return 1.0

def install(agent_module, audit_path):
    PromptAgent = agent_module.PromptAgent
    orig_predict = PromptAgent.predict
    enabled = os.environ.get("CIV_LOOPBREAK", "0") == "1"

    def wrapped_predict(self, instruction, obs):
        cur = obs.get("screenshot")
        nudge = ""
        if enabled and cur is not None and _STATE["prev_png"] is not None:
            frac = _diff_frac(_STATE["prev_png"], cur)
            if frac < 0.005:  # screen unchanged = last action had NO effect
                _STATE["noeffect_streak"] += 1
            else:
                _STATE["noeffect_streak"] = 0
            if _STATE["noeffect_streak"] >= int(os.environ.get("CIV_STREAK","3")):  # loop-breaker trigger (val-tunable)
                _STATE["overrides"] += 1
                nudge = ("\n\n[SYSTEM INTERVENTION] Your last %d actions produced NO visible "
                         "change on screen — they are MISSING their target (menu did not open / "
                         "click landed on nothing). Do NOT repeat the same click. Try a DIFFERENT "
                         "method: use a keyboard shortcut, the menu mnemonic (Alt+key), GIMP's "
                         "action search, or click a clearly different location." % _STATE["noeffect_streak"])
        _STATE["prev_png"] = cur
        _STATE["steps"] += 1
        if nudge:
            instruction = instruction + nudge
        with open(audit_path, "a") as f:
            f.write("step=%d streak=%d override=%s\n" % (_STATE["steps"], _STATE["noeffect_streak"], bool(nudge)))
        return orig_predict(self, instruction, obs)

    PromptAgent.predict = wrapped_predict
    return _STATE
