# Budgeted runs

`SearchRun` manages one synchronous optimization run for `CMAES`, `SHADE`, `NelderMead`, and `RandomSearch` over bounded continuous variables. `PhasedDFO` manages its own budget via `optimize()` and raises `TypeError` if passed to `SearchRun`.

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

`ask()` reserves `optimizer.next_batch_size()` candidates (capped to remaining budget for `RandomSearch`). Resolve each pending batch before calling `ask()` again. `step(objective)` runs ask, evaluate, and tell in place; `minimize()` loops until done and returns `run.result`.

## Budget and failures

One evaluation is one candidate under one repeat. A batch of 32 candidates with `repeats=3` consumes 96 evaluations and selects incumbents by mean score while keeping raw repeat values in `result.raw_results`.

A failing objective, non-finite observation, or mutated candidate tensor stops the run before updating optimizer state. `charged_evals`, `attempted_evals`, `completed_evals`, and `accounting_uncertain` separate billed work from verified observations. Use `EvaluationResult` when external execution returns reordered IDs, explicit completion masks, or repeat matrices of shape `[candidate, repeat]`.

## Planned checkpoints

```python
import torch
from torch_dfo import CMAES, SearchRun

run = SearchRun(CMAES(dim=4, bounds=2.0, seed=7, device="cpu"), max_evals=100)
run.step(lambda x: x.square().sum(-1))
torch.save(run.state_dict(), "run.pt")

state = torch.load("run.pt", weights_only=False)
fork = SearchRun.from_checkpoint(state)
fork.step(lambda x: x.square().sum(-1))
```

Save checkpoints only between completed batches on trusted storage. `from_checkpoint()` creates a new run ID with parent lineage, preserving charged counts and candidate streams on the same device.

## Scope

| Capability | Current behavior |
| --- | --- |
| Methods | `CMAES`, `SHADE`, `NelderMead`, `RandomSearch` |
| Objectives | Batched finite scalar values, minimization |
| Variables | Continuous box bounds (scalar or per-dimension tensors) |
| Repeats | Fixed count, arithmetic mean aggregation |
| Scheduling | Synchronous, one pending batch |
| Devices | CPU, CUDA, and MPS (`float32` on MPS) |
| Checkpoints | Trusted Python payloads at completed batch boundaries |

General constraints, adaptive noisy selection, automatic retries, and distributed scheduling are outside `SearchRun`.
