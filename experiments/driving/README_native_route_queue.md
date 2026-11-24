# Frozen native 100-route queues

`native_route_queue.py` runs the pinned VAD, UniAD and DriveTransformer
configurations on the original 100 Bench2Drive routes and seed 0. It uses their
published checkpoints, six native camera inputs and control implementations,
plus the separately verified logging-only BEV-camera adapter. These methods
have different inputs and planning models from the API driving policy. Their
comparison does not isolate a causal intervention effect.

The controller first waits for the existing GPU setup controller to finish.
It verifies each successful method's original setup result and records its
hash. Methods without a successful setup do not submit routes. A single worker
then waits for one GPU to remain unoccupied for 60 seconds, with at least
36,000 MiB free, available simulator ports, and at least 10 GiB disk space.
Unknown processes are never terminated. A foreign compute process entering
the selected GPU stops only this controller's owned attempt.
An existing GPU setup receipt is checked and kept byte-for-byte with its
original timestamp; restarting does not rewrite this protected input.

`vulkan_devices.c` enumerates device UUIDs through the installed Vulkan loader;
it creates no logical device or GPU allocation. Vulkan adapter indices are
matched to physical NVIDIA UUIDs. After boot, the controller separately checks
the actual Unreal process's GPU before launching native policy inference. A
missing or different mapping is an invalid environment attempt.

Every missing method/route receives a new owned simulator and an immutable
attempt directory. The official evaluator performs the complete agent
constructor, sensor validation, model inference and route lifecycle. No
synthetic forward or setup success is substituted for a route result. The
first route must produce a valid official result before the remaining 99
start; a valid zero passes this technical gate. Exhausting its environment
attempts leaves that method pending for repair. A physical-device identity
mismatch stops the owned queue immediately. The original 4,000-tick
simulation budget is retained. A completed scored task
failure, including TickRuntime, remains valid. Initialization, source, kernel,
transport and simulator errors remain invalid, preserve raw logs and receive
the existing three-attempt budget and 120-second backoff.

Valid raw records and logs are hash protected. Restarts skip scored routes;
old attempts and results are never overwritten. Reports state each method's
own valid denominator, score mean, successes and infractions. Only all 100
valid routes make a method complete. Setup or resource waits are pending,
with zero benchmark scores. No test-score threshold chooses a checkpoint,
starts the next method or changes the frozen configuration.

An optional `--runtime-repair` binds an environment-only simulator device
isolation repair to the unchanged protocol and original attempt budget. The
manifest protects previous invalid logs and the executing source by hash.
It selects the physical PCI bus corresponding to the chosen NVIDIA UUID,
then uses `nvidia-container-cli` to list that GPU's device nodes. A private
Bubblewrap device mount exposes only those NVIDIA nodes and the corresponding
PCI-matched DRM nodes to the owned simulator. The installed Mesa device-select
layer uses `DRI_PRIME` and `MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1`. See the
[Mesa environment documentation](https://docs.mesa3d.org/envvars.html).
The installed layer must first enumerate exactly that UUID as its sole device;
only then does the simulator receive adapter index zero. The native policy's
environment remains unchanged. The existing post-boot check still requires
the owned Unreal process to appear on exactly the selected physical GPU.
Extra or missing devices stop the attempt. This does not relax the identity
guard, add retries, change rendering quality or modify policy inputs.
Mesa enumeration alone is insufficient: an independently recorded boot still
created a small graphics context on another card. The process device namespace
is therefore required as well. Mounts affect only the owned child; host device
nodes and other processes remain unchanged.

## Completed-score boundary

The native actor can remain alive in teardown after writing its final outcome.
`finished_native_record` requires `Finished`, eligibility, progress `[1, 1]`
and the original native record/log validity checks. A complete score, including
zero, is frozen before a subsequent resource check can classify teardown as an
environment failure. Completion is checked again if it occurs during the
resource inventory call. Unfinished episodes retain the original foreign-process
and one-hour no-progress guards. Only the owned actor and simulator are cleaned;
the final native outcome must remain identical after cleanup before its hashes
are committed. This adds no route attempts or policy execution budget.

`recoverable_finished_record` provides a read-only historical qualification rule
for the earlier race: the complete raw record and log must predate the exact
foreign-compute abort. Protocol, runtime and physical-device bindings must be
verified separately. A recovery must apply the same rule to every route,
preserve every invalid marker and later valid attempt, and archive the original
aggregation. The primary record is the earliest qualifying attempt, regardless
of score. Corrected aggregation from old raw files is a historical scoring
re-audit, not a new independent evaluation. Partial, post-abort, wrong-route or
environment-error output cannot be recovered by this rule.

Thirteen offline checks cover zero/positive completion, pending and malformed
records, completion during inventory, unchanged outcome data after cleanup,
historical ordering and the full queue path committing a zero exactly once.
