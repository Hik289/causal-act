# Benchmark setup

Each `.patch` file contains changes needed to connect CausalAct to a benchmark.
Its matching `.revision` file records the exact benchmark commit to use.

| Benchmark | Changes |
| :--- | :--- |
| VisualWebArena | Add refiner calls and action logs, update model support, and fix the numbered marks drawn on screenshots |
| OSWorld | Update model selection and Docker startup settings |
| Bench2Drive | Update how the evaluation code writes results |

Start with a clean benchmark checkout at the recorded commit. Run the
[integration installer](../scripts/install-integration.py) to check and apply
the patch and copy the CausalAct experiment scripts into that checkout. See the
[reproduction guide](../docs/reproduction.md) for the commands for each benchmark.

The benchmark's terms apply to its checkout and patches.
