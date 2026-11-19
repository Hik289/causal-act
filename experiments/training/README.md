# Controlled forum component study

These scripts fill a missing training/evaluation path; they do not turn the
existing replay studies into a complete three-domain experiment.

`build_forum_quads.py` records original, backend-targeted, unrelated, and
translated control states in the running Postmill benchmark. It confirms the
subscription state after reloading: a successful AJAX request can leave a stale
button in the current DOM. Each forum's initial state is saved and restored.
The forum split is fixed before capture. All states and goal paraphrases from
one forum stay in one split. Twenty-five held-out forums provide 100 planned
cases. Failed held-out captures are retained and cannot silently shrink or
replace the test set.

`forum_expert_iteration.py` trains separate A, V, and I modules using the full
SmolVLM-500M backbone with independent rank-eight text-attention adapters and
heads. The full parameter count includes the frozen vision/language backbone;
the executable audit requires each module to remain below one billion
parameters. A predicts click/stop and a region; V judges candidate actions;
I selects one of the three backend-verified executable edit programs.
Intervention selection is constrained to that pool. It is not unrestricted
edit-program generation and its validity rate must be labelled accordingly.

The fixed study uses two V → A → I rounds, three epochs per module per round,
seed 42, and minimum validation-loss checkpoint selection. I's difficulty
targets come from A's errors on training examples; validation labels select
checkpoints, and test cases never enter training or selection. The original,
static counterfactual, A-only, A+I, A+V, three removed-loss variants, and full
configuration use the same immutable dataset. The untrained baseline uses
the pretrained language head's click/stop likelihood, not a random classifier.

The held-out report includes semantic paired/targeted/unrelated accuracy,
region recall, and an oracle decision-risk rate, with numerator and valid
denominator plus area buckets. It does not measure executed pointer success,
real-world safety, Android transfer, or driving success. Empty area buckets
remain unmeasured. Full end-to-end ablations and cross-environment deployment
are separate pending experiments and must not be filled using these scores.

`run_forum_suite.py` schedules the ten configurations on explicitly allocated
GPUs. It waits for verified model/data handoff when configured, preserves
completed reports, stops on partial held-out output rather than rerunning it,
and records worker failures without manufacturing zero scores. Model setup
checks use independent synthetic inputs and are never benchmark results.

`verify_forum_reports.py` reconciles completed predictions to the written metric
definitions without rerunning inference or altering the raw report. It writes
`verified_report.json` with source hashes. PairCausalAcc requires correct
original, targeted, and unrelated decisions; TargetPairAcc counts the first two.
The frozen initial runner counted only the first two under PairCausalAcc, so
existing raw reports must be read alongside this explicit correction.

Run the protocol checks from this directory in the training environment:

```bash
python -m unittest test_forum_suite
```
