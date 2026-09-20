# Independent review of the CMA-ES comparison

September 20, 2026.

## Verdict

Accept the 100 full-budget records as a bounded comparison of the stated configurations. The six-method panel failed because all 20 EvoX runs stopped after 32 evaluations. EvoX has no full-budget quality or time result.

## Evidence checked

- The [saved panel](results/cma-comparison/panel.json) has 120 unique seed, task, and method records. It has 100 successes and 20 EvoX failures.
- Every successful run used 3,200 evaluations and recorded 100 batches. Native `SearchRun` counts agree with the independent counter. All native guard counts are zero.
- Every selected score passes the recorded CPU oracle tolerance. Successful traces keep the best score nonincreasing. Thermal runs record one held-out check per selected candidate.
- The 30 objective probes passed. The largest CPU/CUDA difference was `9.31e-10` on the ellipsoid and passed its relative tolerance. Thermal probe values matched exactly.
- The original thermal admission probes kept their expected decisions. Zero and runaway heat failed. The PI control and constant cooling passed.
- The benchmark, native CMA-ES, thermal evaluator, and covariance-probe hashes in the panel match the reviewed files. The EvoX and pycma wheel source hashes match the pinned upstream files.
- The [plain EvoX workflow probe](probe_evox_workflow.py) reproduced the `torch.cond` aliasing failure on CPU and CUDA without the comparison adapter. This failure is separate from the rank-one covariance defect.

## Interpretation limits

Native CPU had lower median scores than pycma CPU on sphere and rotated Rastrigin. Ellipsoid and thermal train results were mixed. No completed run met a canonical target.

All 25 completed thermal runs met the train threshold. None met the 3.5 held-out diagnostic threshold. That split was already exposed, so this is not new confirmation data.

pycma CPU and pycma with a CUDA objective selected identical final candidates for all 20 seed-task pairs. Their incumbent traces differed only by rounding. Treat them as execution modes of one method for quality comparisons. The other methods use different random streams, even when seed labels match.

CPU was faster for these 32-candidate eager workloads on the measured GB10. Five seeds and one instance per task do not establish general quality or speed rankings. Do not compare these unbounded thermal requests directly with the earlier bounded study.
