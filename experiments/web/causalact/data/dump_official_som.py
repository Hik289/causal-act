"""诊断: 用 VWA 官方 SoM pipeline 渲染一个 forum 页, 对比我手绘的(框不可见)图。
验证 输入输入不对称假设: base 看到官方标注框图, 我喂 A 的是无框图。
"""
from causalact_runtime import location
import os, sys
os.environ.setdefault("DATASET", "visualwebarena")
os.environ.setdefault("REDDIT", "http://127.0.0.1:9999")
os.environ.setdefault("SHOPPING", "http://127.0.0.1:7770")
os.environ.setdefault("CLASSIFIEDS", "http://127.0.0.1:9980")

os.environ.setdefault("WIKIPEDIA", "http://127.0.0.1:8888")
os.environ.setdefault("HOMEPAGE", "http://127.0.0.1:4399")



from browser_env import ScriptBrowserEnv
from PIL import Image
import numpy as np

OUT = location('CAUSALACT_RUN_DIR', 'web/som_diag')
os.makedirs(OUT, exist_ok=True)

env = ScriptBrowserEnv(
    headless=True,
    observation_type="image_som",
    current_viewport_only=True,
    viewport_size={"width": 1280, "height": 720},
)
# minimal config to open /f/AskReddit
import json, tempfile
cfg = {
    "task_id": 99999, "require_login": True, "storage_state": location('CAUSALACT_VWA_AUTH', ''),
    "start_url": "http://127.0.0.1:9999/f/AskReddit",
    "geolocation": None, "intent_template": "", "instantiation_dict": {},
    "intent": "test", "require_reset": False, "eval": {"eval_types": ["string_match"], "reference_answers": {"must_include": ["x"]}},
    "sites": ["reddit"],
}
tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump(cfg, tf); tf.close()
obs, info = env.reset(options={"config_file": tf.name})
som_img = obs["image"]  # numpy array with SoM boxes drawn
Image.fromarray(som_img).save(f"{OUT}/official_som_AskReddit.png")
text_obs = obs.get("text", "")
print("official SoM image saved. text_obs length:", len(text_obs))
print("=== first 600 chars of official text obs (SoM element list) ===")
print(text_obs[:600])
# count elements referencing 'comment'
n_comment = sum(1 for line in text_obs.splitlines() if "comment" in line.lower())
print("lines mentioning 'comment':", n_comment)
env.close()
