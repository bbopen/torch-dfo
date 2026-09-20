# Architecture decision 1: search state and evaluation ownership

Status: revised after independent review. Implementation acceptance still requires the contract tests below.

## Problem

The foundation review reproduced budget overruns in three execution paths, incomplete checkpoints, and false benchmark success. Those failures crossed algorithm, execution, and reporting code. PhasedDFO also calls the objective outside its public ask/tell loop.

A general library needs one owner for actual evaluation accounting. It also needs to expose raw results before a search policy turns them into selection decisions.

## Decision

Build a synchronous `SearchRun` with a single pending batch. Evaluate existing torch-dfo code and directly reusable EvoTorch implementations for each internal adapter. Retain, adapt, or replace code based on tested behavior, maintenance cost, and application needs. Keep legacy exports available while the new interface is evaluated. Do not replace every algorithm at once.

The external interface supports `ask`, `tell`, checkpoint save/load, and inspection of the current result. A `minimize` convenience function drives the same interface. Algorithm adapters cannot call objectives.

The first two algorithm adapters are CMA-ES and random search. They provide a real comparison between atomic and variable batch sizes. Add SHADE after finite-value and failure contracts pass. Keep DLR and the phased scheduler outside the first implementation until the same tests can cover their state transitions.

The objective callable is an injected dependency. The standard tensor evaluator handles deterministic batches, explicit repeats, and microbatching. A simulator adapter uses the same result contract without claiming GPU residency.

## Small public data contracts

`CandidateBatch` holds unique candidate IDs and an owned tensor of candidate values. It also declares the scheduled replicate count and evaluator configuration identity. Keep a private authoritative candidate copy. The returned tensor must not alias that copy or algorithm state. Before managed execution, verify that public candidates still equal the private proposal. Execute on a separate owned copy, then reject any in-place change before accepting its results. External tell must also check the retained public batch against the private proposal. Tell reorders results by ID and updates the algorithm with the original candidates. These checks detect accidental mutation; external evaluation values remain caller-attested, not independently proven.

`EvaluationResult` holds matching IDs, raw objective values, constraint values, completion status, and per-entry attempted-call facts. It preserves one value per candidate and replicate. It does not hide penalties or convert failures into good scores.

Raw objective values have shape `[candidate, replicate]`. Constraints have shape `[candidate, replicate, constraint]`, with `g <= 0` feasible. Status and attempted-call masks match the objective shape. Zero constraint columns is valid. Candidate values remain tensors on the selected device; categorical application adapters use their own representation-aware policy.

The final result exposes the candidate with the best declared aggregate across all completed replicates, total reserved and charged evaluations, raw observations, and a stop reason. The initial aggregate is the arithmetic mean. A single lucky replicate cannot become the incumbent. Call it the best observed aggregate, not the true expected objective. If no valid candidate exists, it explicitly has no incumbent. Do not return an unevaluated zero vector as a solution.

## Budget and transaction rules

The budget unit is one logical evaluation of one candidate under one replicate or scenario. One Python call that evaluates a tensor of 100 candidates counts as 100 evaluations.

The scheduling cap applies to the trusted evaluator supplied by the library. External tell results attest to their counts; the library cannot inspect hidden work inside an arbitrary callback. Label accounting as measured or caller-attested. Record simulator steps, internal oracle work, and post-hoc validation separately. If authoritative attempt facts are lost, charge the full reservation and stop.

An algorithm adapter reports its next required batch size without consuming RNG or changing state. `ask` must fit that batch and all its scheduled replicates inside the available cap. Random search may use a smaller final batch. Atomic CMA-ES generations may leave a visible unused residue.

`used + reserved <= limit` always holds. Asking reserves cost before objective execution. A second ask is invalid until the pending batch resolves. A tell must provide exactly the pending IDs, once each. Reordering is allowed; duplicates, stale IDs, missing entries, wrong shapes, and unknown IDs are rejected before algorithm updates.

Account for evaluation attempts before calling the algorithm update. If the update rejects an otherwise valid result, its objective cost remains charged. Record the raw result and stop reason. Mark the adapter terminal if its update throws, because it may have partially mutated state. Reject further ask, tell, and resumable checkpoint operations. No rollback is claimed without an actual pre-update snapshot.

When an evaluator throws before returning an authoritative call mask, charge the entire reservation conservatively and mark accounting as uncertain. Never refund unknown work or retry it silently. Known unstarted entries can be released only with an explicit executor report. Keep scheduled, charged, known attempted, and completed counts distinct.

The first implementation stops on malformed or nonfinite results. It records the failure and does not inject infinity or an arbitrary penalty into a legacy algorithm. Failure-tolerant updates require a later policy and dedicated tests.

## Checkpoints and randomness

Version one permits checkpoints only at a quiescent completed batch, during a planned pause with no executing or pending work. Reject checkpoint requests after ask and during evaluation. Saving pending proposals is deferred until a durable execution-start journal exists.

Snapshots are not proof that no work occurred after them. Loading an older snapshot must be an explicit fork with a new run identity, not an automatic crash recovery under the old accounting record. Include the checkpoint lineage and retain the original run's cost. Do not replay an interrupted run automatically. Detached jobs and saved result manifests provide the initial network-disconnection fallback.

Exactly-once execution across arbitrary remote failures is not claimed. Later crash-safe resumption must persist a start marker before execution, retain unresolved reservations, and pass a crash-injection test before it can be advertised.

Checkpoint headers include schema, algorithm configuration, bounds and representation, evaluator identity, dtype, device mode, counters, pending state set to none in version one, and seed policy. Validate the header before mutating state. Same-device continuation under the pinned environment must reproduce candidates and IDs exactly. Cross-device migration must be explicit and marked non-bit-exact.

Separate search RNG from evaluator randomness. An evaluator declares either independent per-candidate seeds or common scenario seeds. Controller comparisons need shared held-out scenarios; candidate IDs must not silently select easier environments. Store the seed policy and scenario split in the evaluator fingerprint. Avoid global RNG mutation.

## Constraints and discrete representations

The result contract can represent constraints before every algorithm can use them. Initial scalar adapters reject every nonempty constraint tensor, including all-feasible constraints. Do not describe this as constrained optimization support.

A later constrained policy must preserve raw objective and violation information across generations. Define its comparison and adaptation rules explicitly. Replacing all objective values with ranks would change SHADE's improvement-weighted adaptation and needs its own evaluation.

Do not build categorical search on scalar category order. A discrete adapter must define valid symbols and mutation/sampling rules. Record unique decoded designs separately from evaluations. Caching is opt-in and restricted to a declared deterministic evaluator.

## What stays outside the first implementation

Distributed actor scheduling, general asynchronous batches, Pareto archives, quality diversity, automatic retries, and a replacement PhasedDFO portfolio are deferred. Record requirements that the two real adapters demonstrate before adding them.

Keep application dependencies outside the core PyTorch package. Use an external experiment runner for data acquisition, baseline execution, plotting, and detached Spark jobs.

## Why this design

An enlarged PhasedDFO would continue mixing objectives with search policy. A generic experiment framework would create many interfaces before the need is demonstrated. The chosen design centralizes the contracts that already failed while retaining tested numerical code.

## Acceptance checks

- Count scheduled logical evaluations, known attempts, and conservative charges under tiny, exact, residual, repeated, and failed batches.
- Reject invalid result IDs and shapes without updating the algorithm or losing known cost.
- Mutate public candidates before execution and mutate the executor input during evaluation. Each case must stop before adapter update and preserve the correct charge.
- Restore quiescent planned-pause checkpoints with exact next candidates. Reject pending or poisoned checkpoints. Mark old-snapshot forks with a new run identity and retained lineage.
- Preserve search trajectories across ordered versus reordered results. Match deterministic microbatch results exactly where the kernels permit it; declare numerical tolerances before testing other kernels.
- Prove evaluator seeds do not change with microbatching and do not mutate global RNG.
- Retain every failure and return no incumbent when all evaluations fail.
- Compare total runtime and quality against the direct legacy algorithm and EvoTorch with identical objective work.
- Measure validation, cloning, transfer, and checkpoint overhead. Do not claim a fully GPU-resident loop if host checks synchronize it.
- Before an overhead claim, freeze a gate of no more than 10% median full-run overhead on a representative expensive objective, with an upper 95% interval bound below 15%. Report cheap-objective overhead separately without hiding a failed gate. These proposed margins must survive the runtime smoke before they are frozen.

The architecture is accepted only after an independent challenge and a passing implementation-level contract suite. Application success remains a separate gate.

## Review disposition

The independent reviewer conditionally accepted the direction and required five corrections: budget trust, checkpoint replay, mutable and partially updated state, noisy incumbents, and scoped capability/overhead claims. The rules above address each. The older contracts proposal is exploratory and is superseded where it differs. The fresh review and the implementation-level tests remain separate evidence.

## Library and research lab

The library is the primary deliverable. Its repository will also contain the autonomous research lab, canonical evaluations, engineering applications, and exploratory demonstrations. The lab will exercise the same public interface that library users receive. Tested general improvements return to the library; application dependencies remain optional.

Direct adaptation of EvoTorch source is authorized. A separate implementation is not a requirement. Preserve source provenance, add characterization tests, then measure each proposed improvement. The existing numerical code is useful evidence, not an architectural constraint.
