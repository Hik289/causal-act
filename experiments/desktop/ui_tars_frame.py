"""Map UI-TARS 1.5 processed-image pixels through a recorded service frame.

Frame dimensions come from the deployed processor's image_grid_thw, never from
benchmark target boxes. See ByteDance's README_coordinates.md in UI-TARS.
"""
from functools import wraps
from io import BytesIO
import inspect
import importlib
import math

from s2_repair_runtime import screenshot_image_bytes


class FrameContractMismatch(RuntimeError):
    pass


def checked_size(size):
    if len(size) != 2 or any(not isinstance(x, int) or x <= 0 for x in size):
        raise ValueError("Coordinate frame needs two positive integer dimensions")
    return tuple(size)


def model_to_screen(point, model_size, screen_size):
    mw, mh = checked_size(model_size)
    sw, sh = checked_size(screen_size)
    if len(point) != 2 or not all(math.isfinite(v) for v in point):
        raise ValueError("Grounding point needs two finite coordinates")
    # Preserve out-of-frame predictions; never clamp a model's error onto a target.
    return [round(point[0] * sw / mw), round(point[1] * sh / mh)]


def install_agent_grounding_frame(method, contract):
    """Set an Agent S grounding frame without changing model inference."""
    from PIL import Image
    if method not in ("s2", "s3"):
        raise ValueError("Expected Agent S2 or S3")
    OSWorldACI = importlib.import_module("gui_agents." + method + ".agents.grounding").OSWorldACI

    screenshot_size = checked_size(contract["screenshot_size"])
    model_size = checked_size(contract["model_frame_size"])
    if contract["coordinate_space"] != "qwen25vl_processed_pixels":
        raise ValueError("Expected the UI-TARS 1.5 processed-pixel coordinate space")
    installed = getattr(OSWorldACI, "_causalact_frame_contract", None)
    if installed is not None:
        if installed != contract:
            raise FrameContractMismatch("A different grounding frame is already installed")
        return

    initialize = OSWorldACI.__init__
    signature = inspect.signature(initialize)

    @wraps(initialize)
    def initialize_with_frame(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        params = dict(bound.arguments["engine_params_for_grounding"])
        if params.get("model") != contract["model"]:
            raise FrameContractMismatch("Grounding model differs from the frozen contract")
        params.update(grounding_width=model_size[0], grounding_height=model_size[1])
        bound.arguments["engine_params_for_grounding"] = params
        initialize(*bound.args, **bound.kwargs)
        if (self.width, self.height) != screenshot_size:
            raise FrameContractMismatch("Configured screen differs from the frozen contract")

    generate = OSWorldACI.generate_coords

    @wraps(generate)
    def generate_with_frame(self, referring_expression, obs):
        image = Image.open(BytesIO(screenshot_image_bytes(obs["screenshot"])))
        if image.size != screenshot_size:
            raise FrameContractMismatch("Screenshot size changed; do not guess coordinate scaling")
        return generate(self, referring_expression, obs)

    OSWorldACI.__init__ = initialize_with_frame
    OSWorldACI.generate_coords = generate_with_frame
    OSWorldACI._causalact_frame_contract = dict(contract)


def install_s2_grounding_frame(contract):
    return install_agent_grounding_frame("s2", contract)


def install_s3_grounding_frame(contract):
    return install_agent_grounding_frame("s3", contract)
