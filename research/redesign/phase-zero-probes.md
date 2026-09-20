# Evaluator audit before optimization

Status: planned. None of these application probes has passed yet. Data, runtime, objective direction, and thresholds must be fixed before a search result can be accepted.

## Continuous numerical control

- Degenerate: evaluate the bounds midpoint and a repeated candidate. Do not treat these as independent successful search runs.
- Ceiling: use a synthetic translated sphere with an explicit known optimum to verify objective direction and target arithmetic. Use official COCO target flags on COCO itself.
- Cheap exploit: submit a false negative-infinite result, stale IDs, duplicated IDs, a missing candidate, and a budget overrun. Each must fail the contract without losing attempted cost.
- Null: run random and Sobol designs at the same evaluation count. A family that those solve easily cannot support a hard-problem claim.

Record all probes, including the negative cases. A locally estimated minimum must never become ground truth for a global target.

## MyoSuite Baoding P2 control adapter

- Degenerate: run the zero-action policy on the fixed development scenario list. Save task score, drop outcome, effort, simulator steps, and full episode length.
- Ceiling: obtain a reproducible successful reference trajectory or policy under the pinned native task. If unavailable, leave the quality threshold uncalibrated. Do not declare a ceiling pass from a plausible reward value.
- Cheap exploit: truncate an episode, omit a dropped rollout, repeat only the easiest scenario, or alter reward weights. The evaluator must detect the wrong horizon, missing records, wrong seeds, or wrong task fingerprint.
- Null: compare fixed random policy parameters and random actions under the same scenarios. Record their distributions before freezing a claim threshold.

The initial contract adapter can pass accounting and reproducibility without solving the task. Native Baoding P2 uses CPU simulation. A GPU task requires a separately validated simulator path.

## TFBind8 discrete control adapter

Do not run this adapter until a permitted complete dataset is acquired and hashed. Verify duplicate rows, unique keys, score direction, and native oracle agreement first.

- Degenerate: score one repeated sequence. It must produce one unique design and consume the declared repeated query count.
- Ceiling: the sealed evaluator uses a verified highest-scoring table entry as a private audit fixture. Do not expose that entry to the optimizer or training view.
- Cheap exploit: request an invalid symbol, an incomplete sequence, an absent table key, and an oracle answer without a charged query. The native minimum-score fallback must not conceal missing data.
- Null: compare uniform sequence sampling and direct single-base mutation at the same query count. Report the cost of an exhaustive ranking control separately.

A high score on this fully characterized lookup task is not a new biological discovery or GPU-compute result. Larger surrogate tasks require independent validation beyond their own model scores.

## Frozen record

Freeze only after the relevant probes have observed outcomes. The record must include source/data hashes, allowed training data, held-out boundaries, objective direction, seeds, known cost units, failure rules, and the decision script.

Do not tune an optimizer while repairing its evaluator. Version a corrected evaluator and retain the failed records. No cross-version gain is accepted without rerunning the baseline under the new version.
