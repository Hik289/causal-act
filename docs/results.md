# Experiment results and evidence

This report separates measured outcomes, outcomes reconstructed from recorded counts, and incomplete experiments. See [the summary counts](../results/summary.json) and [the released diagnostic aggregates](../results/web-diagnostics/).

## VisualWebArena: intervention diagnostic and live web execution

- **Targeted state change.** The subscription builder constructs 150 pairs from real, reversible forum subscription changes. The held-out split contains 48 pairs grouped by forum. The base is correct on both members of 0/48 pairs; training on the original only is also 0/48. Training on both states reaches 48/48 with or without the targeted margin. This is a binary click-or-stop diagnostic, not a measurement of general navigation or full four-view causal accuracy.
- **Unrelated changes.** In the text-assisted diagnostic, accuracy under the unrelated edit improves from 1/12 to 10/12 after action supervision; the specificity term yields the same endpoint in that setting. Its extra gain in the image-focused variant was one example on a 12-pair validation set. Treat it as exploratory.
- **Environment checks.** The 264/264 agreement figure pools five types of checks against environment observations and execution, including 60 rendering and 50 executable checks. It does not mean 264 independent edits were audited by people. An early probe falsely passed because a scripted click did not send the subscription request; the corrected test used a real browser click and checked the persisted state.
- **Offline grounding.** With the same Set-of-Mark image and candidate format, the base and trained refiner each hit 89/94 examples overall. On the 26-example forum-disjoint split, the base hit 25/26 and the refiner hit 23/26. A preliminary, much larger apparent gap came from an input mismatch: the base saw rendered marks and descriptions while the refiner's marks were missing.
- **Live web.** The scored forum baseline is 60/161 (37.3%); this excludes tasks without valid in-scope verdicts. A broad-gate pilot fell from 39/100 to 23/100 on the matched subset. A later navigation gate had seven harms and zero rescues among 13 override reruns (−4.1 pp on the reported task pool). The stop gate had one rescue and three harms among 21 override reruns. A later base rerun was still incomplete in the project records; incomplete or lost verdicts do not support a new score.

## OSWorld: independent comparison versus passthrough reconstruction

| Protocol | Denominator | Base | Refiner | Delta |
| :--- | ---: | ---: | ---: | ---: |
| Independent matched outcome files, shared threshold 0.90 | 296 | 75 (25.3%) | 58 (19.6%) | −5.7 pp |
| Passthrough reconstruction, per-domain settings | 300 | 76 (25.3%) | 92 (30.7%) | +5.3 pp |

The independent comparison was checked against the stored per-task result files: 19 failures became successes and 36 successes became failures. The experiment used a 12-step budget and one recorded seed, rather than the full benchmark protocol. The audit records label 17 rescues and four harms as triggered under the shared threshold. A partial base-versus-base rerun changed verdict on 10/46 matched tasks (seven improved, three regressed). This confirms rerun variability on that subset; it does not show that all changes in untouched runs are due to environment noise.

The second line reconstructs a different quantity: retain the Base outcomes on untouched tasks and add the reported net changes from per-domain gate settings (18 rescues − two harms). The underlying per-domain counts come from the study records. The reconstruction is consistent with those counts, but this release does not establish that all 300 episodes were rerun under one precommitted policy, or independently verify the selection split for every threshold. “90%” is 18/(18+2) among **changed outcomes**, not gate precision over all triggers. Identity of a forwarded action on a shared history does not guarantee identical results under independent rollouts.

The earlier GIMP (14 tasks) and VLC (15 tasks) gains were small-sample pilots. They should not replace the broader matched comparison.

## Driving and remaining scope

The Bench2Drive result file contains one Base record (route index 0, Driving Score 0.00879, route completion 0.41%, status “Agent got blocked”). This shows the stack executed and produced a score. The available agent has no verified CausalAct override path; a two-condition driving result is unavailable.

The full 910-task VisualWebArena run, full multi-round expert iteration, comparison against external baselines, and ScreenSpot-Pro and AndroidWorld evaluations were not completed in the inspected records.
