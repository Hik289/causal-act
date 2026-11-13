"""CausalAct driving autoagent for Bench2Drive/CARLA leaderboard.
Design (per README §4.1): BasicAgent = fixed trajectory tracker (route following);
GPT-5.6-sol base = high-level decision (stop/proceed/yield) from front camera;
CausalAct (B+A) adds V-gated high-level decision override at traffic-relevant moments.
CIV_MODE=base -> GPT high-level only; CIV_MODE=causalact -> + V-gated refiner.
"""
from __future__ import print_function
from PIL import Image as _PILImage
import os, io, base64, json, re, urllib.request
import carla
import numpy as np
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from agents.navigation.basic_agent import BasicAgent

def get_entry_point():
    return 'CausalActDrivingAgent'

OPENAI_BASE=os.environ["OPENAI_BASE_URL"]
MODEL=os.environ.get("CIV_DRIVE_MODEL","gpt-5.6-sol")
MODE=os.environ.get("CIV_MODE","base")  # base | causalact
DECISION_EVERY=int(os.environ.get("CIV_DECIDE_EVERY","20"))  # query LLM every N ticks

def _b64_img(arr):
    im=_PILImage.fromarray(arr[:,:,:3][:,:,::-1])  # BGRA->RGB
    b=io.BytesIO(); im.save(b,format="JPEG",quality=70); return base64.b64encode(b.getvalue()).decode()

def _causal_verify(img_b64, base_decision):
    """CausalAct A-module: causal verification of STOP-relevant evidence.
    A second model query checks key evidence the base may miss.
    Overrides base PROCEED/SLOW -> STOP when a red light / pedestrian / stop sign is clearly
    present in the ego lane (reduces unsafe actions = the causal intervention objective)."""
    prompt=("You are a safety verifier for a driving agent. Examine ONLY the key safety evidence in "
            "this front camera image. Is there a RED traffic light for the ego lane, a pedestrian in "
            "or entering the ego lane, or a STOP sign requiring an immediate stop? "
            "Answer ONLY 'danger' (must stop now) or 'clear'.")
    body={"model":MODEL,"max_tokens":2048,"messages":[{"role":"user","content":[
        {"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img_b64}}]}]}
    try:
        req=urllib.request.Request(OPENAI_BASE.rstrip("/")+"/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
        with urllib.request.urlopen(req,timeout=float(os.environ.get("CIV_LLM_TIMEOUT", "30"))) as r:
            txt=json.loads(r.read())["choices"][0]["message"]["content"].strip().upper()
        if re.fullmatch(r"DANGER[.!]?", txt):
            return "STOP"
    except Exception as e:
        print("[CausalAct] LLM err: verifier", type(e).__name__, getattr(e,"code",""), flush=True)
    return base_decision

def _llm_decision(img_b64):
    """Ask GPT for high-level driving decision from front camera."""
    prompt=("You are a driving decision module. Look at this front camera image. "
            "Output ONE word: PROCEED (safe to drive/accelerate), STOP (red light/obstacle/pedestrian/stop sign ahead in ego lane), "
            "or SLOW (caution: yellow light, approaching junction, nearby vehicle). Answer with only the word.")
    body={"model":MODEL,"max_tokens":2048,"messages":[{"role":"user","content":[
        {"type":"text","text":prompt},
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img_b64}}]}]}
    try:
        req=urllib.request.Request(OPENAI_BASE.rstrip("/")+"/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type":"application/json","Authorization":"Bearer "+os.environ["OPENAI_API_KEY"]})
        with urllib.request.urlopen(req,timeout=float(os.environ.get("CIV_LLM_TIMEOUT", "30"))) as r:
            txt=json.loads(r.read())["choices"][0]["message"]["content"].strip().upper()
        action = re.fullmatch(r"(STOP|SLOW|PROCEED)[.!]?", txt)
        if action:
            return action[1]
        print("[CausalAct] Unrecognized decision format", flush=True)
    except Exception as e:
        print("[CausalAct] LLM err:",repr(e)[:100],flush=True)
    return "SLOW"

class CausalActDrivingAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track=Track.SENSORS
        self._agent=None
        self._tick=0
        self._decision="PROCEED"
        self._audit=open(os.environ["CIV_DRIVE_AUDIT"],"a")

    def sensors(self):
        return [{'type':'sensor.camera.rgb','x':0.7,'y':0.0,'z':1.60,'roll':0.0,'pitch':0.0,'yaw':0.0,
                 'width':400,'height':300,'fov':100,'id':'Front'}]

    def run_step(self, input_data, timestamp):
        # init route-following controller (fixed tracker)
        if not self._agent:
            hero=None
            for actor in CarlaDataProvider.get_world().get_actors():
                if actor.attributes.get('role_name')=='hero': hero=actor; break
            if not hero: return carla.VehicleControl()
            self._agent=BasicAgent(hero,30)
            plan=[];prev=None
            for transform,_ in self._global_plan_world_coord:
                wp=CarlaDataProvider.get_map().get_waypoint(transform.location)
                if prev: plan.extend(self._agent.trace_route(prev,wp))
                prev=wp
            self._agent.set_global_plan(plan)
            return carla.VehicleControl()
        # base trajectory-following control
        control=self._agent.run_step()
        # high-level decision from camera every N ticks
        self._tick+=1
        if self._tick % DECISION_EVERY == 0 and 'Front' in input_data:
            img=input_data['Front'][1]
            _img_b64=_b64_img(img)
            d=_llm_decision(_img_b64)
            base_decision = d
            if MODE == "causalact" and d != "STOP":
                d = _causal_verify(_img_b64, d)
            self._decision=d
            self._audit.write(f"tick={self._tick} mode={MODE} base_decision={base_decision} decision={d}\n")
            self._audit.flush()
        # apply high-level decision as override on the tracker
        if self._decision=="STOP":
            control.throttle=0.0; control.brake=1.0
        elif self._decision=="SLOW":
            control.throttle=min(control.throttle,0.3)
        return control

    def destroy(self):
        try: self._audit.close()
        except: pass
