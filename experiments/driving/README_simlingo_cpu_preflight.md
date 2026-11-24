# SimLingo CPU checkpoint preparation

`simlingo_cpu_preflight.py` constructs the pinned native Hydra architecture and
strictly loads the published flat checkpoint on CPU. It retains the native
tokenizer additions, image configuration, LoRA settings and model heads. The
two native pretrained-model loads resolve to an already downloaded, hashed
local snapshot. Source, checkpoint, configuration, base assets and dependency
versions are bound before construction; CUDA initialization and HTTP requests
are rejected during the check.

The check compares every loaded state tensor to its checkpoint value in the
constructed model's dtype. It records parameters, tensor keys, devices and
runtime versions. It executes no native agent setup, forward pass, simulator
or route. `valid=true` describes architecture/checkpoint preparation and has
no benchmark denominator or task score.

The recorded November 24 preparation used the pinned native source
`743b243afd6cf5ff51b9fa1f8cac86f22d569684`, the published epoch 013 checkpoint,
and InternVL2-1B base revision `0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187`.
Strict loading matched all 992 state tensors, with 957,195,520 model parameters
on CPU. No checkpoint or parameter was selected using held-out driving scores.

An isolated environment reused existing Torch 2.4.1 and torchvision 0.19.1;
the upstream environment specifies 2.2.0 and 0.17.0. NumPy 1.23.0 was installed
inside the new environment while the parent environment remained unchanged.
The inherited, unused MMCV package reports a NumPy metadata conflict there;
this environment is scoped to SimLingo CPU preparation. FlashAttention was
not installed or executed. These differences and complete package versions
are retained in the preparation evidence. GPU kernels, native agent lifecycle
and Bench2Drive evaluator compatibility remain separate pending checks.

A separate CPU import check accepted the native four-sensor description through
the target Bench2Drive sensor validator and checked setup, step and destroy call
signatures. The camera remains 1024×512 with a 110-degree field of view. This
checks the import/schema boundary between the source's Bench2Drive 0.0.3 setup
and the target 0.0.4 API; it does not verify their complete route lifecycle.

Create a private protocol with:

- `scope=cpu_checkpoint_preparation`, `method=SimLingo`;
- native source root, commit and relative file hashes;
- checkpoint/configuration paths and their published hashes;
- the pinned base-model asset manifest path and hash, repository and revision;
- the executing preflight file hash;
- the reused Torch version/path, upstream Torch version and exact package map.

Run it in the registered CPU environment:

```bash
python simlingo_cpu_preflight.py \
  --protocol /absolute/path/cpu_protocol.json \
  --protocol-sha256 FROZEN_PROTOCOL_SHA256 \
  --output /absolute/path/new_preparation_attempt
```

The output directory must be new. Failures retain a null score and traceback;
completed preparation is preserved rather than repeated. Subsequent GPU
readiness must share the existing native simulator resource/ownership limits
and retain the original method's inputs and control behavior. Preparation
does not append SimLingo to another frozen method queue.
