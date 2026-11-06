"""OSWorld docker-provider smoke: 起 VM + 截图 + 一个动作 + success 判定。
验证 Desktop 环境可用 (kvm 解锁后)。输出 -> /data2。
"""
from causalact_runtime import location
import os, sys, json
os.environ.setdefault("SCREEN_WIDTH", "1280")
os.environ.setdefault("SCREEN_HEIGHT", "720")
sys.path.insert(0, location('CAUSALACT_OSWORLD_DIR', ''))
from desktop_env.desktop_env import DesktopEnv
from PIL import Image
import numpy as np, io

OUT = location('CAUSALACT_RUN_DIR', 'stop_full_results/osworld_smoke')
os.makedirs(OUT, exist_ok=True)

# 一个简单 chrome task 做 smoke
task_path = None
import glob
for p in glob.glob(location('CAUSALACT_OSWORLD_DIR', 'evaluation_examples/examples/chrome/*.json')):
    task_path = p; break
cfg = json.load(open(task_path))
print("smoke task:", cfg.get("id"), "|", cfg.get("instruction", "")[:70], flush=True)

print("init DesktopEnv (docker provider)...", flush=True)
env = DesktopEnv(provider_name="docker", action_space="pyautogui",
                 screen_size=(1280, 720), os_type="Ubuntu",
                 require_a11y_tree=False, headless=True)
print("env created. reset to task...", flush=True)
obs = env.reset(task_config=cfg)
print("reset OK. obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs), flush=True)
# screenshot
scr = obs.get("screenshot") if isinstance(obs, dict) else None
if scr is not None:
    if isinstance(scr, bytes):
        Image.open(io.BytesIO(scr)).save(f"{OUT}/screenshot_init.png")
    else:
        Image.fromarray(np.array(scr)).save(f"{OUT}/screenshot_init.png")
    print("saved screenshot_init.png", flush=True)
# one action (noop-ish: move mouse)
try:
    obs2, reward, done, info = env.step("import pyautogui; pyautogui.moveTo(640, 360)")
    print("step OK. reward:", reward, "done:", done, flush=True)
    scr2 = obs2.get("screenshot") if isinstance(obs2, dict) else None
    if scr2 is not None:
        (Image.open(io.BytesIO(scr2)) if isinstance(scr2, bytes) else Image.fromarray(np.array(scr2))).save(f"{OUT}/screenshot_after.png")
        print("saved screenshot_after.png", flush=True)
except Exception as e:
    print("step error:", str(e)[:200], flush=True)
# success eval
try:
    score = env.evaluate()
    print("evaluate() score:", score, flush=True)
except Exception as e:
    print("evaluate error:", str(e)[:200], flush=True)
env.close()
print("SMOKE_DONE", flush=True)
