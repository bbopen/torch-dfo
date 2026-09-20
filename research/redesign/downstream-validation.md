# Downstream validation

Date: 19 September 2026.

## Product decision

Build an independent, maintained PyTorch derivative-free optimization library for engineering and research. Public applications identify useful workloads and requirements. Test task results, current installation, numerical behavior, reproducibility, and documentation. Measure acceleration separately.

The thermal control study did not show a held-out engineering advantage. Keep that result. It does not decide whether the library can serve other applications.

EvoTorch is one reference implementation and benchmark baseline. Include relevant alternatives from other stacks, following `reference-strategy.md`. Drop-in compatibility, equivalent configuration objects, and migration tooling are not requirements. torch-dfo keeps its own API and capability roadmap. Adapt application objectives to that API when benchmarking.

## Upstream maintenance evidence

GitHub's API reports that the latest default-branch commit is `cebcac4f20979078becf8b908016cd8e5e6714a4`, dated 14 May 2025. Release `v0.6.1` has the same date. The repository is not archived. Seven pull requests remain open, including contributions from 2026. A recent repository push timestamp therefore does not establish a recent default-branch change.

Describe the dates directly. Visible release and default-branch activity are limited; contributions continue. These observations do not establish the maintainers' future intentions.

Sources: [commit](https://github.com/nnaisense/evotorch/commit/cebcac4f20979078becf8b908016cd8e5e6714a4), [release](https://github.com/nnaisense/evotorch/releases/tag/v0.6.1), [open contributions](https://github.com/nnaisense/evotorch/pulls).

## Public application sequence

| Application | Existing use | What to evaluate | Scope |
| --- | --- | --- | --- |
| [calisim](https://github.com/Plant-Food-Research-Open/calisim/tree/fa6539b2d1e4a39210fef3d9dabee97b893ec69c) | Fits simulation parameters through EvoTorch engines | Simulation discrepancy, parameter recovery where identifiable, held-out simulation predictions, evaluations, integration effort | Reproduce a small continuous calibration case first. Mixed and categorical variables require separate support. |
| [EvoIK](https://github.com/knowledgetechnologyuhh/evo_ik/tree/090ad201d821d9043764c6ad180e180d3a3fdcfe) | Uses batched forward kinematics with CMA-ES | Position error, orientation error, joint limits, solve rate, evaluation count, time | Reproduce the published robot fixture. This tests a real robot workload; include gradient-based IK before claiming an advantage over other IK methods. |
| [Contimask](https://github.com/eth-siplab/contimask/tree/f536b4e8946adf529ec58330dbc4207c4f321ec3) | Uses PGPE to optimize neural perturbation masks | Prediction change, mask size and recovery of known synthetic features, runtime | Start with released synthetic data. PGPE and neural parameter evaluation are current capability gaps. The source includes thresholded masks; compare gradient methods only with disclosed objective differences. |
| [DynAMO](https://github.com/michael-s-yao/DynAMO/tree/52dfc8fe4cb4de6374635bc5f971974f027d4d1f) | Uses CMA-ES and Cosyne within offline model-based design | True oracle quality and candidate diversity, alongside surrogate score and cost | Requires datasets and surrogate training. Keep training cost separate and report it. A better surrogate score alone does not pass. |
| [MyoChallenge](https://github.com/PKU-MARL/MyoChallenge/tree/feef99336d2b09f6b8a57462ade30b3902017512) | Uses PGPE and ClipUp for muscle-driven hand policies | Task success across new rollout seeds, environment steps, wall time | Follow-on neuroevolution case. Simulator parallelism does not establish GPU acceleration. |

These are observed source integrations. Publication results and deployment claims have not been independently reproduced.

## First bounded reproduction

Run calisim and EvoIK first. Limit each initial dependency investigation to two installation attempts. Preserve exact failures and source revisions.

1. Run an unmodified upstream entry point or test where possible.
2. Label any repair separately and preserve the original result.
3. Freeze task inputs, objective, tolerances, and search budgets before comparing optimizers.
4. Check the evaluator with a known solution, an unrelated solution, and invalid or out-of-domain candidates.
5. Use the same objective for an optimizer substitution. Count all candidate evaluations.
6. Recheck final candidates using domain quantities, not only optimizer status.
7. Report successful runs and failures. Keep partial reproduction separate from paper replication.

A reproducible incompatibility is an investigation result. It is not a successful application test or evidence that torch-dfo resolves the problem.

## Decision after reproduction

Select a representative application suite from observed needs. Avoid adding algorithms merely to lengthen the catalog.

For application value, execute a documented torch-dfo workflow on a supported stack. For comparisons, use a shared objective, environment, and resource budget where possible. Record unavoidable stack differences and each execution outcome. Predeclare task tolerances and budgets. Require a documented torch-dfo workflow that executes a substantive task and meets those tolerances. Small application adapters are acceptable. This need not beat EvoTorch's score. A small synthetic calibration smoke test alone cannot establish application value.

For claims of an advantage over domain methods, include strong domain baselines. For optimizer or acceleration comparisons, freeze the named baselines, independent task instances, evaluation budgets, and complete timing. Report the distribution across seeds. Include setup, data transfer, and synchronization in end-to-end timings.

Preserve a separate research track for new applications. Its experimental results must not silently become library guarantees.

## Active phase

Turn the calibration smoke into a native torch-dfo benchmark with SciPy differential evolution and optional EvoTorch as baselines. Use counted objective calls, repeated seeds, and a fixed exploratory validation segment. All 21 observations were exposed during earlier exploration; this is not an untouched confirmation set. Audit the evaluator before search. Freeze the protocol before reading comparative results. This is a workload adaptation, not a calisim backend implementation.
