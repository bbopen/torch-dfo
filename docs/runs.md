# Budgeted runs

`SearchRun` manages one synchronous optimization run. It supports CMA-ES, SHADE, and random search on bounded continuous variables.
The objective returns one scalar per candidate. Lower is better.

## Evaluate batches

```python
from torch_dfo import CMAES, SearchRun

run = SearchRun(CMAES(dim=4, bounds=2.0, seed=7, device="cpu"), max_evals=100)
while (batch := run.ask()) is not None:
    values = batch.x.square().sum(-1)
    run.tell(batch, values)

result = run.result
print(result.best_value, result.charged_evals, result.stop_reason)
```

`ask()` reserves a batch before evaluation. Resolve that batch before asking for another.
`step(objective)` performs ask, evaluate, and tell. `minimize` repeats those steps and returns the final result.
Both `step` and `tell` update the run in place. Read `run.result` when you need a detached snapshot.

For an external simulator, evaluate the rows of `batch.x` and pass their values to `tell`.
The caller reports that work. The library cannot inspect hidden simulator calls.
Keep simulator steps and validation work in the experiment record.

## Budget and failures

One evaluation means one candidate under one repeat. A tensor call with 32 rows counts as 32 evaluations.
CMA-ES and SHADE need complete generations. Random search can consume a partial final batch.

With `repeats=3`, the managed evaluator calls the objective three times per batch.
The mean of those three values selects the incumbent. Raw results preserve each observation.
The objective owns its simulation randomness. Use explicit fixed scenarios or an independent generator for reproducible comparisons.

A failed objective stops the run. Unknown work consumes its full reservation; the run does not retry automatically.
Nonfinite values and changed candidate tensors also stop the run before an algorithm update.
`charged_evals`, `attempted_evals`, `completed_evals`, and `accounting_uncertain` distinguish cost from available evidence.
A run with no valid observation has `best_x=None` and `best_value=None`.

Use `EvaluationResult` when external execution returns reordered IDs, repeat observations, or explicit completion masks.
Its values have shape `[candidate, repeat]`. The simple `tell(batch, values)` form also accepts a vector when `repeats=1`.

## Planned checkpoints

```python
import torch
from torch_dfo import CMAES, SearchRun

run = SearchRun(CMAES(dim=4, bounds=2.0, seed=7, device="cpu"), max_evals=100)
run.step(lambda x: x.square().sum(-1))
torch.save(run.state_dict(), "run.pt")

# Load only a checkpoint you trust. This beta stores a Python optimizer object.
state = torch.load("run.pt", weights_only=False)
fork = SearchRun.from_checkpoint(state)
fork.step(lambda x: x.square().sum(-1))
```

Save only between completed batches. Pending or failed runs cannot produce a resumable checkpoint.
Loading creates a new run identity and records its parent in `lineage`.
It preserves the candidate stream and charged work at that checkpoint.
Work performed after the snapshot still belongs to the parent run; add branch costs when comparing total research effort.

Continuation requires the same device and compatible Python, PyTorch, and torch-dfo versions.
These checkpoints support a planned pause. They do not provide automatic recovery of an interrupted external evaluation.

## Beta limits

| Capability | Current scope |
| --- | --- |
| Methods | CMA-ES, SHADE, uniform random search |
| Objectives | Batched finite scalar values; minimization |
| Variables | Continuous box bounds |
| Repeats | Fixed count, arithmetic mean |
| Scheduling | Synchronous, one pending batch |
| Devices | CPU and CUDA validation; legacy MPS support requires float32 |
| Records | Detached tensors and counters; benchmark runner adds source hashes and JSON output |
| Checkpoints | Trusted Python payloads, planned pauses, same-device continuation |

General constraints, mixed-variable search policies, automatic retries, and distributed scheduling are outside this interface.
The legacy `SearchSpace`, `PhasedDFO`, and `DFOOptimizer` interfaces remain available with their own documented behavior.

Raw observations stay in memory for the run. Result snapshots copy them.
For long studies, use bounded runs and save their reports between trials.
