"""Runner for Agent S2 on OSWorld."""
import os
import json

from gui_agents.s2.agents.agent_s import AgentS2
from gui_agents.s2.agents.grounding import OSWorldACI
from gui_agents.s2.core.engine import OpenAIEmbeddingEngine

# Monkeypatch embedding engine to be a dummy
def dummy_embed(*args, **kwargs):
    return [0.0]*1536
OpenAIEmbeddingEngine.get_embeddings = dummy_embed

import run as runmod


class AgentS2Wrapper:
    def __init__(self, action_space="pyautogui", observation_type="a11y_tree", **kwargs):
        self.action_space = action_space
        self.observation_type = observation_type

        screen_width = int(os.environ.get("OSW_WIDTH", 1920))
        screen_height = int(os.environ.get("OSW_HEIGHT", 1080))

        engine_params = {
            "engine_type": "openai",
            "model": os.environ["CAUSALACT_BASE_MODEL"],
            "base_url": os.environ["OPENAI_BASE_URL"],
            "api_key": os.environ["OPENAI_API_KEY"]
        }

        engine_params_for_grounding = {
            "engine_type": "openai",
            "model": "ui-tars-1.5-7b",
            "base_url": os.environ["CAUSALACT_GROUNDER_URL"],
            "api_key": os.environ.get("CAUSALACT_GROUNDER_KEY", "unused")
        }

        grounding_agent = OSWorldACI(
            platform="ubuntu",
            engine_params_for_generation=engine_params_for_grounding,
            engine_params_for_grounding=engine_params_for_grounding,
            width=screen_width,
            height=screen_height,
        )

        self.agent = AgentS2(
            engine_params=engine_params,
            grounding_agent=grounding_agent,
            platform="ubuntu",
            action_space=action_space,
            observation_type=observation_type,
            search_engine=None,
            embedding_engine_type="openai",
            memory_folder_name=os.environ["CIV_S2_MEMORY_DIR"],
        )

    def predict(self, instruction: str, obs: dict):
        import numpy as np
        import base64
        import io
        from PIL import Image

        cur_obs = obs.copy()
        scr = cur_obs.get("screenshot")
        if isinstance(scr, np.ndarray):
            img = Image.fromarray(scr)
            bio = io.BytesIO()
            img.save(bio, format="PNG")
            cur_obs["screenshot"] = base64.b64encode(bio.getvalue()).decode("utf-8")
        elif isinstance(scr, bytes):
            cur_obs["screenshot"] = base64.b64encode(scr).decode("utf-8")

        res_dict, actions = self.agent.predict(instruction, cur_obs)
        # Agent S2 predicts string actions for pyautogui

        # Remove massive base64 images from res_dict to keep logs clean
        clean_res = {}
        for k, v in res_dict.items():
            if isinstance(v, str) and len(v) > 1000:
                clean_res[k] = "<long_string>"
            else:
                clean_res[k] = v
        return json.dumps(clean_res), actions


    def reset(self, *args, **kwargs):
        self.agent.reset()

runmod.PromptAgent = AgentS2Wrapper

args = runmod.config()
with open(args.test_all_meta_path, "r") as f:
    test_all_meta = json.load(f)
if args.domain != "all":
    test_all_meta = {args.domain: test_all_meta[args.domain]} if args.domain in test_all_meta else test_all_meta
test_file_list = runmod.get_unfinished(args.action_space, args.model, args.observation_type, args.result_dir, test_all_meta)
runmod.test(args, test_file_list)
print("[AgentS2] DONE", flush=True)
