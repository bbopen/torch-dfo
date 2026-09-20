# EvoTorch PGPE reuse review

Reviewed EvoTorch commit `cebcac4f20979078becf8b908016cd8e5e6714a4` against `torch-dfo-redesign` commit `5b2c03c1c2d14ebd38f22946526fb0f8d3a3dcd3`. No source code was changed.

## Recommendation

Adapt, do not take EvoTorch as a runtime dependency. A new diagonal-Gaussian PGPE adapter is worthwhile for high-dimensional continuous parameter search when the objective is an expensive, batched PyTorch computation. It has linear state and work in the parameter dimension. Full CMA-ES stores and decomposes a dense covariance matrix. The current CMA-ES path explicitly builds and updates that matrix at `torch_dfo/cmaes.py:533-576` and recomputes its eigensystem at `:596-632`.

Do not graft PGPE before the redesign contracts exist. The active decision starts with `SearchRun`, `CandidateBatch`, `EvaluationResult`, and adapters for CMA-ES and random search. It defers new algorithms until their result, failure, budget, and checkpoint rules have tests. See `research/redesign/architecture-decision.md:11-19`, `:21-45`, and `:77-90`. No `SearchRun` implementation exists in the current source tree.

When that contract lands, add a PGPE adapter behind it. The adapter must only produce candidates and consume returned results. It must not call an objective, mutate a public candidate tensor, bypass pending-batch accounting, replace raw objective values with ranks outside its own update, or hide non-finite results. Those are contract invariants, not optional PGPE details.

## Smallest code reuse boundary

EvoTorch's functional PGPE has a clean algorithm boundary. `pgpe_ask` samples from a diagonal Gaussian around a centre. `pgpe_tell` computes score-function gradients from the same candidate tensor and fitness vector, updates the centre via a chosen optimiser, then bounds the standard-deviation update. See `evotorch/src/evotorch/algorithms/functional/funcpgpe.py:301-318` and `:330-384`.

Use this as the reference implementation, then extract a torch-dfo-native module rather than importing EvoTorch's package. The target should have one state object with `mean`, `sigma`, `velocity`, `generation`, RNG state, and immutable configuration. Keep the target API class-shaped to fit `BaseOptimizer` and later adapter ownership. `BaseOptimizer` owns device, dtype, population storage, best tracking, and serialisation now. See `torch_dfo/base.py:13-65` and `:82-163`.

The direct EvoTorch source chain is `funcpgpe.py` to `funcclipup.py`, `functional/misc.py`, `decorators.py`, `distributions.py`, `tools/ranking.py`, and `tools/misc.py`. `funcpgpe.py` imports the vmap decorator, both diagonal distributions, functional sampler and gradient factories, `modify_vector`, and optimiser dispatch. The distribution calculation calls `rank`. Any direct copy must carry those symbols or replace them with torch-dfo code. The copied mathematical kernel needs only these pieces:

1. Antithetic diagonal sampling. EvoTorch requires an even population and emits `mean + noise` and `mean - noise` pairs. See `distributions.py:616-668`.
2. Rank-to-utility conversion for a minimisation objective, then the symmetric PGPE mean and standard-deviation estimators. EvoTorch routes fitness through `rank` before gradient calculation at `distributions.py:236-299`. Its symmetric distribution centres non-centred utilities before calculating gradients at `:708-755`.
3. A sigma rule. Preserve a positive lower bound, an optional upper bound, and a maximum fractional change per update. EvoTorch applies that rule at `funcpgpe.py:368-371`; its generic helper is `tools/misc.py:868-900`.
4. A safe centre optimiser. The first implementation should expose either momentum SGD or guarded ClipUp. Do not bring the string-dispatch layer for Adam and SGD from `functional/misc.py:26-75` into the first adapter.

That leaves out `Problem`, `SolutionBatch`, Ray, Gym, logging, and EvoTorch's general distribution framework. The public EvoTorch import currently also needs Ray, which this environment lacks. A direct dependency would turn a single-Torch library into a dependency on `cma`, `gymnasium`, plotting packages, Pandas, and `ray>=1.0`. See `evotorch/setup.cfg:27-44`.

## ClipUp correction

EvoTorch ClipUp sets the velocity to momentum plus a unit gradient step and clips the velocity norm. See `funcclipup.py:94-108`. I reproduced the exact body in an isolated Torch script. The zero vector gave velocity `[nan, nan, nan]`. A finite `[3, 4, 0]` gradient gave `[0.06, 0.08, 0.0]` and finite state. `[inf, 1, 0]` also produced a non-finite state.

The torch-dfo adaptation must reject or stop on a non-finite gradient. For a finite zero gradient, use `momentum * velocity` and then apply the usual velocity cap. It must never divide by zero. Tests must cover zero, non-finite, and finite gradients. The contract layer should stop before the adapter update for non-finite evaluator output, as the architecture decision already requires at `architecture-decision.md:41-45`.

## Reuse choices

**Adapt, recommended.** Extract and adapt the relevant diagonal PGPE and ClipUp source into a small torch-dfo module. Preserve the visible algorithm behaviour that tests establish. This fits the redesign's minimal PyTorch dependency and its controlled adapter boundary.

**Copy with local dependencies.** Copy `funcpgpe.py`, `funcclipup.py`, the needed rank logic, symmetric sampler, `modify_vector`, and the `expects_ndim` batch wrapper. This keeps the source structure but brings a large and coupled slice of EvoTorch. It also needs a deliberate rewrite of the old `functorch` fallback. Torch 2.10 has `torch.func`.

**Keep EvoTorch as a dependency.** This provides the complete framework quickly, but it conflicts with torch-dfo's runtime dependency policy and duplicates its evaluator ownership. It is unsuitable for the core library.

## Tests and attribution for a future graft

Freeze the following checks before comparing against CMA-ES or SHADE:

1. Samples have shape `[population, dimension]`, stay on the configured device, and use an even population in antithetic mode.
2. Same seed gives the same candidates and next state on one device. Cross-device continuation stays explicitly non-bit-exact, matching the existing base policy at `torch_dfo/base.py:89-132`.
3. Finite sphere and shifted-ellipsoid objectives improve versus seeded random search at the same candidate-evaluation count. This is an acceptance comparison, not a GPU claim.
4. Finite zero fitness remains valid. NaN and Inf fitness stop through the result contract before PGPE state changes. A zero update gradient must preserve finite state.
5. A public candidate mutation, result-ID mismatch, reordered result set, checkpoint boundary, and exact budget residue follow the redesign contract tests. See `architecture-decision.md:77-88`.

Keep a source header in each adapted implementation and add a README credit when code lands. Point to EvoTorch commit `cebcac4f20979078becf8b908016cd8e5e6714a4`, the copied source paths `src/evotorch/algorithms/functional/funcpgpe.py` and `funcclipup.py`, and the Apache-2.0 licence in `work/evotorch/LICENSE`. State that torch-dfo adapted the code and changed its ownership and finite-value policy.
