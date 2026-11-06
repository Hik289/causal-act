# CausalAct: Controlled Visual Interventions for Reliable Agent Actions

*Research note · November 2026*

A visual agent can select the wrong action because it misses an important state change, or because it reacts to a visual change that leaves the task unchanged. CausalAct turns these cases into replayable tests: change the application state, keep the task fixed, and ask an environment checker what can legitimately be done. In a forum subscription task, clicking Subscribe advances the goal if the user is unsubscribed. If the same user is already subscribed, stopping is correct. A background change should not reverse that choice; moving the control should change where the agent clicks.

Each original state can be paired with a targeted edit, an unrelated edit, and a layout edit. The evaluator checks actions against acceptable sets and compares semantically aligned actions across views. This distinction matters: changing a feature does not require a different action when a shared acceptable action remains. When an edit is meant to change real application state, its rendered observation alone does not establish that it worked.

The implemented pipeline combines scripted interventions, a lightweight action refiner, and an optional verifier gate. Training uses environment labels where available. At execution, the base action passes through when the gate does not propose a correction. The gate's benefit depends on both how often it rescues an error and how often it replaces a correct base action.

The most controlled test is a forum-disjoint subscription diagnostic: original-plus-edited correctness rises from 0/48 for the base to 48/48 after paired supervision. Equal-weight supervision on both states also reaches 48/48, so the experiment isolates a benefit of supplying the counterfactual data but does not show an additional benefit from a targeted margin term. The live web navigation experiment is negative. On desktop, a shared-threshold independent comparison is negative, while a per-domain reconstruction that reuses the base outcomes on untouched tasks is positive. These estimates have different protocols and should not be used interchangeably.

The artifact does not establish full-benchmark improvements across domains. The forum and desktop comparisons use one recorded seed; the desktop runs use a 12-step budget. The available driving implementation is a Base smoke test. Full expert iteration, comparative driving evaluation, and broad external baseline replication were not completed.

See [the results audit](results.md) for exact denominators and [the reproduction guide](reproduction.md) for the runnable entry points.
