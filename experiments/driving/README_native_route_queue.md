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
