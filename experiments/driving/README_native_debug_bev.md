# Native driving agents without the logging BEV camera

The pinned VAD, UniAD and DriveTransformer agents add a `bev` RGB camera at
50 m when `IS_BENCH2DRIVE` is enabled. Their official SENSORS validator rejects
its extrinsics. This camera supplies the disabled recording path. It is
separate from the model's learned BEV features.

`native_debug_bev.without_debug_bev(module, expected_source_sha256)` produces
an isolated subclass. The caller supplies the hash of its frozen native
agent. The adapter checks that `SAVE_PATH` is `None`, logging calls require
it to be enabled, and the camera is not consumed by the control path. It
removes exactly one camera conversion and one returned debug field from the
native `tick` method, and excludes the exact known 50 m camera from `sensors`.
Other tick statements remain in their original order. It does not rewrite
the native file, add a replacement image, or override `run_step`.

Unexpected source changes, camera specifications, debug dependencies or
enabled logging are errors. The recording method also fails closed. The
caller must still run the unmodified official sensor validator; no validator
rule is bypassed.

For a frozen evaluator entry point, import the native module, construct the
adapter with its recorded hash, expose the returned class, and return that
class's binding name from `get_entry_point()`. Keep the original published
configuration, checkpoint, sensor images, inference and control settings.
Record the adapter's `adapter_audit`, native and adapter source hashes,
environment differences and runtime identifiers alongside the original run.

The compatibility checks include a CPU fixture on each actual native module:
the original and adapted tick functions receive identical six-camera,
GPS, IMU and speed data. Every non-debug output must agree, while the adapted
function receives no debug image. The original sensor list must be reported
separately from the adapted list. These checks perform no model forward,
simulator evaluation, training, threshold selection or benchmark scoring.
Checkpoint compatibility, GPU kernels, complete agent lifecycle and live
route evaluation remain separate gates.

Run the dependency-free checks with:

```sh
PYTHONPATH=experiments/driving python -m unittest experiments/driving/test_native_debug_bev.py
```

Python 3.8 is supported; the adapter uses structural AST operations rather
than `ast.unparse`.
