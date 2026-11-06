"""Trajectory-shared passthrough refiner (v4).

Eliminates LLM run-to-run nondeterminism confound: on NON-triggered steps, replay the
base run's recorded `response` for this exact task+step (re-parse to actions), so ours is
action-identical to base until the gate fires. Only when gate triggers (>=STREAK frozen
screens) do we call the LLM with the nudge -> clean divergence point.

Gate: >=CIV_STREAK consecutive screen-frozen steps (base is in a doom loop). Because up to
the trigger step ours == base, any task where the gate never fires is byte-identical to
base => hurt impossible there. Hurts can only come from genuine post-trigger divergence,
which is the real causal effect we want to measure.
"""
from causalact_runtime import location
import os, io, json, glob
import numpy as np
from PIL import Image, ImageChops

BASE_ROOT = os.environ.get("CIV_BASE_ROOT",
    location('CAUSALACT_RUN_DIR', 'desktop_base_gimp_results/pyautogui/screenshot/gpt-5.6-sol'))
_S = {"prev": None, "streak": 0, "ov": 0, "steps": 0, "task": None,
      "base_resp": None, "idx": 0, "triggered": False}

def _frz(a, b):
    try:
        ia = Image.open(io.BytesIO(a)).convert("RGB"); ib = Image.open(io.BytesIO(b)).convert("RGB")
        if ia.size != ib.size: ib = ib.resize(ia.size)
        return float((np.asarray(ImageChops.difference(ia, ib)).sum(2) > 20).mean())
    except Exception:
        return 1.0

def _load_base(task):
    # find base task dir (domain subdir unknown; glob)
    for p in glob.glob(f"{BASE_ROOT}/*/{task}"):
        tj = p + "/traj.jsonl"
        if os.path.exists(tj):
            rows = [json.loads(l) for l in open(tj)]
            return [r.get("response", "") for r in rows]
    return None

def install(agent_module, audit_path):
    PA = agent_module.PromptAgent
    orig = PA.predict
    en = os.environ.get("CIV_REPLAY", "1") == "1"
    STREAK = int(os.environ.get("CIV_STREAK", "4"))
    cap = int(os.environ.get("CIV_CAP", "3"))

    def wrapped(self, instruction, obs):
        cur = obs.get("screenshot")
        tid = getattr(self, "_civ_task", None)
        # task boundary reset
        if tid != _S["task"]:
            _S["task"] = tid; _S["streak"] = 0; _S["idx"] = 0; _S["triggered"] = False; _S["prev"] = None
            _S["base_resp"] = _load_base(tid) if (en and tid) else None; _S["_taskov"] = 0
        # frozen-screen streak
        frozen = False
        if en and cur is not None and _S["prev"] is not None:
            frac = _frz(_S["prev"], cur)
            if frac < 0.002: _S["streak"] += 1
            else: _S["streak"] = 0
            frozen = _S["streak"] >= STREAK
        _S["prev"] = cur; _S["steps"] += 1

        trigger = en and frozen and _S["ov"] < 10**9 and not _S["triggered"] and _S.get("_taskov", 0) < cap
        i = _S["idx"]; _S["idx"] += 1

        if en and (not trigger) and (not _S["triggered"]) and _S["base_resp"] is not None and i < len(_S["base_resp"]):
            # REPLAY base response verbatim -> action-identical passthrough
            resp = _S["base_resp"][i]
            # keep bookkeeping consistent: orig predict appends to observations inside
            # message-building; on replay we must mirror that so the
            # len(observations)==len(actions)==len(thoughts) assertion holds.
            try:
                from mm_agents.agent import encode_image
                b64 = encode_image(cur) if cur is not None else None
            except Exception:
                b64 = None
            self.observations.append({"screenshot": b64, "accessibility_tree": None})
            try:
                masks = None
                actions = self.parse_actions(resp, masks)  # appends to self.actions
            except Exception:
                actions = None
            if not actions:
                # parse_actions already appended an (empty) entry to self.actions, and we
                # appended one observation. Roll BOTH back before falling back to a real
                # LLM step (orig will re-append obs+action+thought) to avoid desync.
                try: self.observations.pop()
                except Exception: pass
                try: self.actions.pop()
                except Exception: pass
                with open(audit_path, "a") as f:
                    f.write(f"task={tid} step={_S['steps']} streak={_S['streak']} mode=replay_fallback_llm ov=False\n")
                return orig(self, instruction, obs)
            self.thoughts.append(resp)
            with open(audit_path, "a") as f:
                f.write(f"task={tid} step={_S['steps']} streak={_S['streak']} mode=replay ov=False\n")
            return resp, actions

        if trigger:
            _S["ov"] += 1; _S["triggered"] = True; _S["_taskov"] = _S.get("_taskov", 0) + 1
            nudge = ("\n\n[SYSTEM] You have repeated the same ineffective action many times with ZERO "
                     "change on screen - you are stuck in a dead loop and will fail. Abandon this approach "
                     "NOW. Use a completely different method: keyboard shortcut (Ctrl/Alt+key), menu mnemonic "
                     "(Alt+letter) then arrows+Enter, or the application command search. Do NOT click the same spot.")
            with open(audit_path, "a") as f:
                f.write(f"task={tid} step={_S['steps']} streak={_S['streak']} mode=TRIGGER ov=True\n")
            return orig(self, instruction + nudge, obs)

        # after trigger: free LLM (diverged branch) OR base exhausted -> normal LLM
        with open(audit_path, "a") as f:
            f.write(f"task={tid} step={_S['steps']} streak={_S['streak']} mode=llm ov=False\n")
        return orig(self, instruction, obs)

    PA.predict = wrapped
    # reset per-task override counter inside task boundary
    return _S
