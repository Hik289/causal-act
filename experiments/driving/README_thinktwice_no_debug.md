# ThinkTwice without its recording camera

The [pinned native agent](https://github.com/OpenDriveLab/ThinkTwice/blob/e9cf2fc078f6ab8175b1a4929faaa8be55f3aa97/leaderboard/team_code/thinktwice_agent.py)
includes a 50 m `topdown` camera for recording. It also assigns a recording
folder variable that is undefined when `SAVE_PATH` is unset.

`thinktwice_no_debug.without_recording_camera(module, expected_source_sha256)`
creates a separate subclass after verifying the original source hash and
disabled recording. It removes exactly that camera and its tick conversion
and output, and sets the unused recording folder field to `None`. It rejects
unexpected dependencies or enabled logging. The original source is never
rewritten. Model construction, checkpoint loading, the four 1600×900/FOV150
RGB inputs, LiDAR history, navigation and native `run_step` and control methods
retain their original implementation.

This adapter is separate from the six-camera VAD/UniAD/DriveTransformer
adapter. ThinkTwice's remaining sensors are four RGB cameras, LiDAR, IMU,
GNSS and speed. The caller must still run the unmodified official SENSORS
validator, native dependency checks, GPU setup/forward and complete simulator
lifecycle before registering any formal route evaluation. Record the original
source/checkpoint/config identities and any Python or CARLA version differences.

Offline checks verify disabled-recording setup, temporal data preservation,
unchanged control method identity and rejection of extra debug dependencies.
A separate check against the complete pinned source compares four native tick
frames and every non-debug output, including merged LiDAR history. That check
explicitly stubs heavy imports, model setup and image channel conversion;
it is not a native dependency, GPU, sensor-validator or simulator test.
No benchmark result is implied by these checks, and no active queue is changed.

```sh
python -m unittest discover -s experiments/driving -p 'test_*debug*.py' -v
```

The source adapter uses structural AST operations and supports Python 3.8.
