import os
import io
import json
import numpy as np
from PIL import Image
from gui_agents.s3.agents.agent_s import AgentS3
from gui_agents.s3.agents.grounding import OSWorldACI
import run as benchmark

from pathlib import Path
import runpy
from gui_agents.s3.core import engine as s3_engine

class EmptyPlannerResponse(BaseException):
    pass

class BoundedOpenAI(s3_engine.OpenAI):
    def _make_status_error(self, err_msg, *, body, response):
        error = body.get("error", body) if isinstance(body, dict) else {}
        if (str(response.request.url).startswith(os.environ["OPENAI_BASE_URL"] + "/")
                and response.status_code == 502
                and isinstance(error, dict)
                and error.get("code") == "empty_completion"):
            raise EmptyPlannerResponse("Opus returned an empty completion after transport retries")
        return super()._make_status_error(err_msg, body=body, response=response)

s3_engine.OpenAI = BoundedOpenAI
run_example = benchmark.lib_run_single.run_single_example

def run_example_checked(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    try:
        return run_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores)
    except EmptyPlannerResponse as error:
        (Path(example_result_dir) / "invalid_inference.txt").write_text(str(error) + "\n")
        raise RuntimeError(str(error)) from None

benchmark.lib_run_single.run_single_example = run_example_checked

environment = None
original_environment = benchmark.DesktopEnv

def capture_environment(*args, **kwargs):
    global environment
    environment = original_environment(*args, **kwargs)
    return environment

class AgentS3Adapter:
    def __init__(self, action_space="pyautogui", observation_type="a11y_tree", **kwargs):
        self.action_space = action_space
        self.observation_type = observation_type
        self.agent = None

    def reset(self, *args, **kwargs):
        if self.agent is not None:
            self.agent.reset()

    def predict(self, instruction, observation):
        state = dict(observation)
        screenshot = state["screenshot"]
        if isinstance(screenshot, np.ndarray):
            image = Image.fromarray(screenshot)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            state["screenshot"] = buffer.getvalue()
        if self.agent is None:
            planner = dict(engine_type="openai", model=os.environ["CAUSALACT_BASE_MODEL"],
                           base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"])
            grounder = dict(engine_type="openai", model="ui-tars-1.5-7b",
                            base_url=os.environ["CAUSALACT_GROUNDER_URL"], api_key=os.environ.get("CAUSALACT_GROUNDER_KEY", "unused"))
            with Image.open(io.BytesIO(state["screenshot"])) as image:
                width, height = image.size
            grounding = OSWorldACI(env=environment, platform="linux",
                                   engine_params_for_generation=planner,
                                   engine_params_for_grounding=grounder,
                                   width=width, height=height)
            self.agent = AgentS3(worker_engine_params=planner, grounding_agent=grounding,
                                 platform="linux")
        info, actions = self.agent.predict(instruction, state)
        cleaned = {key: value for key, value in info.items()
                   if not isinstance(value, str) or len(value) < 2000}
        return json.dumps(cleaned, default=str), actions

benchmark.PromptAgent = AgentS3Adapter
benchmark.DesktopEnv = capture_environment
args = benchmark.config()
with open(args.test_all_meta_path) as handle:
    tasks = json.load(handle)
if args.domain != "all":
    tasks = {args.domain: tasks[args.domain]}
worker = Path(args.result_dir).parent
invalid_scores = runpy.run_path(str(Path(__file__).with_name("s3_score_validity.py")))["invalid_scores"]
for score in invalid_scores(worker):
    recorded = score.with_name(f"result.invalid.{score.stat().st_mtime_ns}.txt")
    score.replace(recorded)
unfinished = benchmark.get_unfinished(args.action_space, args.model,
                                      args.observation_type, args.result_dir, tasks)
benchmark.test(args, unfinished)
