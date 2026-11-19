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

The separate `balanced_followup_worker.py` and `balanced_followup_controller.py`
implement one pre-registered correction for Static CF and Action Refiner. The
replayed training stream contained 1,587 click and 333 stop labels. Only the
action cross-entropy is reweighted by inverse training frequency; region and
consistency losses remain unchanged. A single-example loss is explicitly
multiplied, since mean-reduced class weights would cancel for batch size one.

This followup keeps the original 160 training and 40 validation cases, and
preselects 80 new cases from the remaining 20 unused forums. These cases are
clustered paraphrases, not 80 independent forums. The original 100-case results
remain intact. The new run waits for the original ten configurations to finish,
then selects checkpoints on validation only. A held-out comparison runs only
when validation PairCausalAcc improves and the sum of targeted/unrelated
accuracy does not decrease. The frozen original and adjusted checkpoint share
the new cases; the unchanged pretrained baseline is also measured. An unmet
validation gate is retained as a negative outcome with no held-out score.

The operational controller expects an immutable run directory with `code/`,
`data/`, and `protocol.json`; copy these scripts into that run's `code/`.
The protocol supplies the original run/model/Python paths, derived class
weights, and validation gate. Existing reports and partial predictions are
never overwritten to restart a comparison.

Run the protocol checks from this directory in the training environment:

```bash
python -m unittest test_forum_suite test_balanced_followup
```
