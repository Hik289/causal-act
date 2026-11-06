# Upstream integrations

Each `.patch` is relative to the revision in its companion `.revision` file.

| Integration | Change |
| :--- | :--- |
| VisualWebArena | Refiner hooks and action audit, model adapter and Set-of-Mark drawing fixes |
| OSWorld | Model dispatch and Docker startup configuration |
| Bench2Drive | Leaderboard result writer compatibility |

Use [the integration installer](../scripts/install-integration.py) with a clean checkout at the recorded revision. The installer checks before applying a patch and adds the corresponding project scripts. Upstream terms apply to each patch and checkout.
