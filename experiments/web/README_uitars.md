# UI-TARS offline preparation

This prepares a future external Forum configuration. It is not a complete
UI-TARS agent or an end-to-end benchmark result. Real model inference, task
reference delivery, browser actions beyond the synthetic fixture, official
evaluation and queue integration still require validation before scoring.

The native source is [ByteDance UI-TARS](https://github.com/bytedance/UI-TARS),
pinned at `582f3a7ea5d285ee8ed9e2e84048d1ab01453c49`. Its unmodified
`codes/ui_tars/action_parser.py` performs Qwen2.5-VL postprocessing. The local
model service uses fp16 across two GPUs and thumbnails each image to a maximum
side of640. A1280×720 screenshot becomes640×360, then the deployed processor's
`image_grid_thw=[1,26,46]` establishes a644×364 model coordinate frame. This
geometry comes from source and processor metadata, without task targets or
benchmark scores.

`uitars_contract.py` runs with `CUDA_VISIBLE_DEVICES` empty. It refuses a
thumbnail cap that disagrees with the service source, checks native-parser and
processor geometry, records source/configuration hashes and streams hashes of
local checkpoint files. The weight manifest identifies files on disk; it does
not prove the running service loaded those exact bytes. The script never calls
the service, loads model weights, changes its image policy or modifies a live
evaluation source. Existing E8 single-image and S2 frame protocols stay separate.

`uitars_contract_fixture.py` uses the actual pinned native parser and a real
Chromium page to click a synthetic center button and check retained DOM state.
A second scripted prediction outside the frame must remain outside after
mapping; coordinates are not clamped. The fixture makes zero model requests,
creates no benchmark score and is excluded from every experimental denominator.

The three active Forum evaluation workers retain their slots. Preparation
does not start a fourth model-evaluation worker. A real inference fixture and
formal UI-TARS queue require an available slot and a completed native adapter;
offline preparation must not be reported as completed external evaluation.
