# Experiment results

This page explains the recorded results and how they were calculated. The
[summary file](../results/summary.json) contains the counts, and the
[diagnostic results](../results/web-diagnostics/) contain the saved test scores.

A **rescue** is a task that fails with the base agent and succeeds with the
refiner. A **regression** is a task that succeeds with the base agent and fails
with the refiner. The saved result files call regressions `harms`. A change of
one percentage point is written as 1 pp.

## VisualWebArena

### Subscription state test

The data builder creates 150 pairs by changing actual forum subscription state.
Each change can be reversed. The test set contains 48 pairs from forums kept
out of training. Each pair contains the original and edited state, and the model
chooses between clicking the marked control and stopping.

The same 500M-parameter model gets both states correct in 0/48 pairs before training.
Training on original states alone also gives 0/48. Training on both states gives
48/48, with or without the extra targeted loss term. This measures whether the
model reacts correctly to subscription state. General navigation and tests with
all four views require separate evaluation.

### Unrelated appearance changes

When the input includes text descriptions as well as the image, accuracy on
unrelated edits improves from 1/12 to 10/12 after action training. Adding the
extra loss for stability under unrelated changes gives the same final accuracy
in that setting. In the version focused on image input, that loss improves
accuracy by one example on a 12-pair validation set. This is an early result
from a small sample.

### Environment checks

The 264/264 agreement count combines five kinds of checks against environment
observations and execution. It includes 60 rendering checks and 50 execution
checks. Several checks can apply to an edit, so the total counts checks.

An early check incorrectly passed even though a scripted click had not sent the
subscription request. The corrected check used a real browser click and
confirmed that the application saved the new state.

### Finding the right target

Both models received the same screenshot with numbered targets, called
**Set-of-Mark**, and the same candidate format. The base and trained refiner
each selected the correct target in 89/94 examples. On the 26 examples from
forums kept out of training, the base scored 25/26 and the refiner scored 23/26.

An earlier comparison appeared to show a much larger gap. The inputs differed
in that comparison. The base saw numbered marks and descriptions, while the
refiner's image was missing the marks.

### Running tasks in the browser

The scored forum base run completed 60/161 tasks (37.3%). This count includes
tasks with a valid result within the available forum setup.

An early test that allowed corrections in many situations fell from 39/100
successful tasks to 23/100 on the same task set. A later navigation check had
seven regressions and zero rescues among 13 reruns with action replacements,
giving a reported drop of 4.1 pp across the task pool. The stop check had one
rescue and three regressions among 21 reruns with action replacements.

A later base rerun was still incomplete in the project records. A new success
rate requires the missing or unfinished task results.

## OSWorld

The verifier scores how likely the base agent is to fail. A chosen score
threshold helps decide when to intervene. The study records contain one
comparison with a shared threshold and another calculation with a threshold for
each application group.

| Calculation | Tasks | Base successes | Refiner successes | Change |
| :--- | ---: | ---: | ---: | ---: |
| Separate runs on matching tasks, one shared threshold of 0.90 | 296 | 75 (25.3%) | 58 (19.6%) | −5.7 pp |
| Calculation from recorded base outcomes and per-domain corrections | 300 | 76 (25.3%) | 92 (30.7%) | +5.3 pp |

### Separate runs on matching tasks

The stored result files show 19 rescues and 36 regressions. These runs used
12 steps per task and one random seed. A full benchmark comparison would require the
official evaluation settings.

The records label 17 rescues and four regressions as tasks where the verifier
triggered a correction at the shared threshold. A partial rerun of the base
agent changed the result on 10/46 matching tasks. Seven failures became
successes and three successes became failures. This shows that rerunning the
base can change results on that sample. The source of each change requires
further checks.

### Calculation that keeps base outcomes

The second row keeps the recorded base result for tasks where no correction
was made. It then adds the reported changes at the thresholds selected for each
domain. This gives 76 base successes + 18 rescues − 2 regressions = 92 successes.

This calculation matches the recorded counts. A comparison based on 300 complete
reruns under a single fixed policy would need further evidence. The available
records also leave the validation split used to select every domain threshold
unverified.

The reported 90% is 18/(18+2), the fraction of changed task outcomes that
improved. Measuring precision across all verifier triggers would also require
the outcomes that stayed the same. Forwarding the same action from a saved
history preserves that action, while separate runs can later diverge.

The earlier gains on GIMP (14 tasks) and VLC (15 tasks) came from small pilot
runs. The broader comparison above covers more tasks.

## Driving

The Bench2Drive result file contains one base-agent run on route index 0.
Its Driving Score is 0.00879, route completion is 0.41%, and the status is
"Agent got blocked." The run confirms that the setup executed and produced
a score. A verified path for applying CausalAct driving corrections and a
comparison between the two agents remain to be completed.

## Experiments still pending

The inspected records leave the following work incomplete.

- The full 910-task VisualWebArena run.
- Repeated rounds of intervention generation and model training.
- Comparisons with external baselines.
- ScreenSpot-Pro and AndroidWorld evaluations.
