# CausalAct: Controlled Visual Interventions for Reliable Agent Actions

*Research note · November 2026*

A visual agent can miss a state change that matters to the task. It can also
change its action because of a detail that has no effect on the task. CausalAct
studies both problems by editing a scene, keeping the goal fixed, and using the
environment to check which actions are correct.

## A simple example

Consider the task "Subscribe to this forum." If the user has not subscribed, the
agent should click Subscribe. If the user is already subscribed, it should stop.
Changing the page background should preserve the decision. Moving the control
should change where the agent clicks.

We test three kinds of edits.

| Edit | What changes | What we check |
| :--- | :--- | :--- |
| Task state (targeted) | A state that matters to the goal, such as whether the user is subscribed | The action is correct for the new state |
| Unrelated appearance | A detail such as a decorative background, with the task state preserved | The agent stays correct |
| Layout | The position of the same control | The action reaches the control at its new position |

For each state, the environment records a set of acceptable actions. An edit can
leave the same action valid in both states. State checks and execution checks
confirm that an edit changed the application as intended.

## Training and action checks

The current code uses scripted edits to create training examples. A small action
refiner learns from the original and edited scenes. An optional verifier checks
when a correction should replace the base agent's proposed action. This check is
called a **gate** in the code.

If the gate keeps the original action, the system forwards the base proposal
directly. A correction can turn a failed task into a success, or turn a success
into a failure. We count both outcomes when assessing the verifier.

## Work completed

The clearest controlled test uses the same 500M-parameter model before and after
training on forum subscription states. The training and test sets contain
different forums. Before training, the model gets both states correct in 0 of
48 test pairs. After training on both states, it gets both correct in all 48.
Adding the extra targeted loss term gives the same result.

The live web navigation experiment reduced task success. On desktop, independent
runs used one shared score threshold to decide when to intervene and reduced
success. A separate calculation uses a threshold for each application group and
keeps the recorded base outcomes for tasks without an intervention. It gives a
positive estimate. The [results report](results.md) explains how each number was
obtained.

The forum and desktop experiments used one recorded random seed. Desktop runs allowed
12 steps per task. Driving code supports a short base-agent test. The remaining
work includes the full benchmark runs, repeated cycles of intervention
generation and training (expert iteration), comparisons with external methods,
and a driving comparison with CausalAct enabled.

See the [reproduction guide](reproduction.md) for setup and commands.
