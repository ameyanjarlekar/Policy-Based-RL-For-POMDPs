# Policy-Based RL Algorithms for POMDPs

This repository contains the paper and experimental Python scripts associated
with **"Scalable Policy-Based RL Algorithms for POMDPs"**, accepted at NeurIPS
2025. The paper focuses primarily on theoretical results; its appendix includes
two small experimental studies. The scripts here also include additional
experiments on toy partially observable Markov decision processes (POMDPs).

## Experiments

### Noisy FrozenLake

The paper's Appendix C.1 studies how observation noise and finite history length
affect performance in `FrozenLake-v1`. The environment is non-slippery, and
each observation is replaced by a different state with probability `p`. The
agent represents its input using the most recent `k` observation-action pairs,
estimates action values with TD(0), and updates a tabular policy using POLITEX
(exponentiated-gradient updates). The paper reports that longer histories can
mitigate noise and presents results across multiple values of `p` and `k`.

[`Frozen-lake.py`](./Frozen-lake.py) implements one particular setting:
`p=0.3`, `k=3`, and five million training episodes by default, with a policy
update every 1,000 episodes. It plots episode rewards and a 1,000-episode
moving average. It does not run the full noise/history sweep shown in the
paper.

### Finite-history comparisons on toy POMDPs

The paper's Appendix C.2 describes a two-state, two-action, two-observation
POMDP and compares learning with history lengths 1 and 2 over 200 episodes of
20 steps. It describes tabular policies updated greedily from Q-values.

The toy-POMDP scripts in this repository are related explorations, but they
should not be treated as exact reproductions of that appendix experiment:

- [`combined_evals.py`](./combined_evals.py) defines a stochastic two-state
  POMDP, uses history lengths 1 and 2, and plots learning curves for one-step
  updates and a 20-step return. It runs 2,000 episodes per setting, with 20
  steps per episode. Its plot labels the 20-step curve as "Cayci et al."; the
  implementation and settings should be checked against the paper before
  interpreting it as a faithful baseline reproduction.
- [`comparison_with_semih.py`](./comparison_with_semih.py) uses a different
  toy POMDP with noisy observations, and compares softmax policy updates using
  one-step and 5-step TD returns. It uses a history length of 2, 200 episodes,
  and 30 steps per episode.
- [`change_finite_history.py`](./change_finite_history.py) uses that same
  noisy toy environment to compare one-step and 30-step TD policy updates
  across history lengths 1, 2, 3, and 4. It runs 250 episodes per setting,
  with 30 steps per episode.

These scripts report episode rewards and plot learning curves; they do not
perform statistical aggregation across independent random seeds.

## Files

| File | Purpose |
| --- | --- |
| [`NeurIPS-2025-scalable-policy-based-rl-algorithms-for-pomdps-Paper-Conference.pdf`](./NeurIPS-2025-scalable-policy-based-rl-algorithms-for-pomdps-Paper-Conference.pdf) | NeurIPS 2025 paper, including the theoretical results and Appendix C experimental descriptions. |
| [`Frozen-lake.py`](./Frozen-lake.py) | Noisy-observation FrozenLake experiment using a history-based tabular Q-function and POLITEX policy updates. |
| [`change_finite_history.py`](./change_finite_history.py) | Toy-POMDP comparison of one-step and 30-step TD policy updates for several history lengths. |
| [`comparison_with_semih.py`](./comparison_with_semih.py) | Toy-POMDP comparison of one-step and 5-step TD policy updates at history length 2. |
| [`combined_evals.py`](./combined_evals.py) | Toy-POMDP experiment comparing one-step and 20-step-return Q updates at history lengths 1 and 2; generates learning curves. |
| [`plot.py`](./plot.py) | Summary and sampled-learning-curve helper functions intended to operate on a `results` mapping. It has no main routine and neither imports NumPy nor creates or loads `results`; a caller must provide those dependencies before using the helpers. |

## Running the experiments

Run scripts from the repository root with Python. For example:

```bash
python change_finite_history.py
python comparison_with_semih.py
python combined_evals.py
python "Frozen-lake.py"
```

The scripts open Matplotlib plots. `Frozen-lake.py` defaults to five million
episodes and has no command-line options; edit the arguments to `train()` in
its main block to change the episode count, history length, or observation
noise. The scripts require Python with NumPy and Matplotlib; the FrozenLake
script additionally uses Gym and tqdm. No dependency/version lockfile is
provided, and the FrozenLake script uses the classic Gym `step()` return
signature, so a compatible Gym version may be needed.
