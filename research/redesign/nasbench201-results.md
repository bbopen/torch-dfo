# NASBench201: search quality and CPU/CUDA execution

torch-dfo CMA-ES and SHADE improved on uniform random search on all three datasets.
They did not establish a general advantage over categorical aging evolution or EvoX DE.
For this small float64 workload, CPU was faster in every eager configuration.
Compiled EvoX DE's steady generations also favored CPU.

These are separate findings. The quality panel completed 720 searches and retained
180 EvoX CMA-ES failures. The timing panel completed 90 eager searches and 30 compiled
EvoX DE searches. It retained 30 EvoX CMA-ES timing failures.

## Workload and fairness

Use the [frozen protocol](nasbench201-protocol.md) for exact settings.
The official EvoXBench NASBench201 database contains 15,625 encoded architectures.
Each has six categorical edges and two or three recorded training trials at fidelity 200.
Search observes one sampled validation error per proposal. Every repeat costs one observation.
Only the selected architecture receives a final mean test score, after all searches finish.

The quality panel uses 30 seeds and a budget of 1,000 observations per run on each dataset.
The continuous optimizers use population 20 and 30 random-key coordinates in `[-1, 1]`.
A fixed, score-blind category permutation resolves exact ties for each seed and edge.
A private architecture-and-visit hash supplies identical trial draws across methods.
Uniform random search samples categories directly. Aging evolution uses population 100
and tournament size 25, with one-edge mutations and oldest-member removal.

The comparator is stock EvoX 1.4.0 at commit `1c242cc9533fbd7e2e0c0342456f4e0f52449e5d`.
No EvoX algorithm was patched. Native CMA-ES and EvoX CMA-ES retain different latent
boundary updates. SHADE and EvoX DE are different algorithms. These are implementation
comparisons under a common evaluation budget, not claims of identical optimizer mathematics.

## Search quality

Mean selected test error, in percentage points, across 30 runs. Lower is better.

| Method | CIFAR-10 | CIFAR-100 | ImageNet16-120 |
| --- | ---: | ---: | ---: |
| torch-dfo CMA-ES, CPU | 5.665 | 27.018 | 53.646 |
| torch-dfo CMA-ES, CUDA | 5.695 | 26.948 | 53.724 |
| torch-dfo SHADE, CPU | 5.650 | 26.979 | 53.490 |
| torch-dfo SHADE, CUDA | 5.661 | 27.076 | 53.594 |
| EvoX DE, CPU | 5.691 | 26.978 | 53.467 |
| EvoX DE, CUDA | 5.753 | 27.039 | 53.704 |
| EvoX CMA-ES, CPU | 30/30 failed | 30/30 failed | 30/30 failed |
| EvoX CMA-ES, CUDA | 30/30 failed | 30/30 failed | 30/30 failed |
| Uniform categorical random | 5.959 | 27.309 | 54.017 |
| Categorical aging evolution | 5.627 | 26.907 | 53.677 |

Aging evolution selected the same architecture in all 30 runs for each CIFAR dataset.
The native methods often found it too, but less consistently. EvoX DE had the lowest
mean ImageNet16-120 error in its CPU configuration. This does not establish a reliable
lead over native SHADE; the result is close and the runs have variation.

The native CPU methods reduced mean error versus random by 0.29 to 0.31 points on CIFAR-10,
0.29 to 0.33 on CIFAR-100, and 0.37 to 0.53 on ImageNet16-120.
The [analysis](results/nasbench201/analysis.json) includes medians, quartiles, all method
failure counts, and paired-seed bootstrap intervals. Those intervals are descriptive
and have no multiple-comparison correction.

Repeated designs matter. Native CMA-ES visited about 235 to 258 unique architectures per
1,000 observations, SHADE about 346 to 402, and uniform random about 970.
Repeated draws can lower an observed validation minimum without improving the network.
The [metric probes](results/nasbench201/evaluator-probes.json) expose that possible shortcut.
Counting repeats and reporting separate test scores prevents presenting the shortcut
as a free improvement.

EvoX CMA-ES failed after 20 observations in every run on the tested stack.
Its `UncapturedHigherOrderOpError` prevents a valid score or speed ranking here.
These failures remain in every denominator. They are not poor optimization scores.

## Execution on DGX Spark

The timing panel uses the same data, encoding, trial schedule, population, dtype, and
budget. It uses seeds 0 through 4 on every dataset. The environment is NVIDIA GB10,
PyTorch 2.13.0+cu130, and one CPU thread. The core library code is unchanged.

All methods use a common device-resident validation table. This removes the Python
lookup and transfer costs from the quality runner. It is an adapted EvoXBench scorer,
not the stock Django API. Every architecture, trial value, and counted trace was audited.
The study does not train or time the selected networks.

The table shows median eager managed-search time for all 1,000 observations.
It includes the scorer and normal optimizer loop. Table staging and optimizer
construction are recorded separately. A ratio above one means CUDA took longer.

| Dataset | Method | CPU ms | CUDA ms | CUDA / CPU time |
| --- | --- | ---: | ---: | ---: |
| cifar10 | torch-dfo CMA-ES | 13.63 | 74.90 | 5.50x |
| cifar10 | torch-dfo SHADE | 10.99 | 47.71 | 4.34x |
| cifar10 | EvoX DE | 4.58 | 13.48 | 2.94x |
| cifar100 | torch-dfo CMA-ES | 13.66 | 74.67 | 5.47x |
| cifar100 | torch-dfo SHADE | 11.02 | 48.42 | 4.39x |
| cifar100 | EvoX DE | 4.60 | 13.57 | 2.95x |
| ImageNet16-120 | torch-dfo CMA-ES | 13.73 | 75.36 | 5.49x |
| ImageNet16-120 | torch-dfo SHADE | 10.96 | 47.57 | 4.34x |
| ImageNet16-120 | EvoX DE | 4.60 | 13.60 | 2.95x |

Objective-only timing also favored CPU: about 1.8 ms versus 5.4 ms for 1,000 lookups.
The separate cheap-objective runs measure optimizer and loop overhead.
Both measurements are retained in the raw timing artifact. These populations are
small, the objective is a cheap lookup, and all arithmetic uses float64.
This panel does not test large populations, float32, or expensive tensor simulations.

### Compiled EvoX DE

Compiled CUDA reduced median time per steady generation by 2.08 times relative to
its own eager CUDA path. This comparison divides each duration by its generation count:
49 eager generations and 48 compiled steady generations. Compiled CPU remained faster than compiled CUDA.
The next table covers the 48 generations after initialization and first compiled use.
It is not a complete cold-search time or a comparison against a compiled native solver.

| Dataset | CPU ms, 48 generations | CUDA ms, 48 generations | CUDA / CPU time |
| --- | ---: | ---: | ---: |
| cifar10 | 2.841 | 6.204 | 2.18x |
| cifar100 | 2.887 | 6.225 | 2.16x |
| ImageNet16-120 | 2.888 | 6.271 | 2.17x |

The first compiled generation cost 3.66 seconds on CPU and 2.58 seconds on CUDA in
panel order. These numbers combine tracing, compilation, and one generation.
Later workflows reuse the disk compiler cache and have separate recorded first-use costs.
Every compiled workflow resets its in-process cache, captures one graph, and retains
that graph through the steady generations. This avoids silent eager fallback.

Preparing the shared trial schedule cost 5.28 to 5.86 seconds per dataset and seed.
Both devices reuse that schedule. Warm search timings exclude this setup cost.
The precomputed schedule controls evaluation noise; it is not a production requirement.

## Evidence and corrections

The independent source audit compared 140,625 validation values and 46,875 test means
against the official EvoXBench evaluator. Validation values matched exactly.
The largest test-mean difference was 2.84e-14 from arithmetic order.

The independent quality audit checked all 723,600 observations, including failed runs.
It verified trial choices, visits, incumbents, counts, and all 720 final test selections.
The independent timing audit checked 120,000 observations with zero value differences.
All 30 compiled searches and six compiled warmups captured exactly one graph each.

| Attempt | Finding | Decision |
| --- | --- | --- |
| Fixture preflight | CUDA device alias and short repeat-probe schedule mismatch | Fix before measurement; 12 final focused tests passed |
| Source staging | Missing Git metadata, then macOS archive metadata made the upstream checkout dirty | Restore the pinned clean checkout before search |
| Quality panel | 720 complete runs; 180 stock EvoX CMA-ES runtime failures | Retain every run; no parameter tuning |
| First timing panel | Compiler cache limit allowed possible eager fallback | Reject compiled speed labels; retain artifact and value audit |
| Corrected timing panel | Full-graph compilation and graph reuse verified | Retain results; record 30 stock EvoX CMA-ES failures |

The complete Spark regression passed 1,164 tests, with 15 skips and one expected failure.
Coverage collection was disabled for this isolated container. Ruff lint and formatting pass.
This change adds benchmarks and documentation. It does not alter optimizer implementations.

## Reproduce

Fetch the pinned official source and data outside the repository:

```sh
bash research/redesign/nasbench201-acquire.sh /path/to/nas-data
python benchmarks/nasbench201_data.py \
  --database /path/to/nas-data/database/db.sqlite3 --output /path/to/nas-data
/path/to/nas-data/venv/bin/python benchmarks/nasbench201_upstream_audit.py /path/to/nas-data
python benchmarks/nasbench201_probes.py \
  --validation-file /path/to/nas-data/search-validation-hp200.json --output probes.json
```

Install a CUDA-enabled PyTorch build and NumPy for the benchmark runtime.
Use a clean checkout of EvoX at the pinned commit. Its package dependencies are optional
for torch-dfo; they are not added to the core installation.

```sh
PYTHONPATH=src python benchmarks/nasbench201.py \
  --validation-file /path/to/nas-data/search-validation-hp200.json \
  --test-file /path/to/nas-data/final-test-hp200.json \
  --evox-source /path/to/evox --output quality.jsonl

PYTHONPATH=src python benchmarks/nasbench201_device.py \
  --validation /path/to/nas-data/search-validation-hp200.json \
  --evox-source /path/to/evox --compile-evox --output device.json

python benchmarks/nasbench201_summary.py \
  --quality quality.summary.json --device device.json --output analysis.json
```

Use empty compiler-cache directories for the initial timing pass. Run the two panels
sequentially. The quality command returns a nonzero exit code when comparator runs fail.
Inspect its saved summary before deciding whether a run needs correction.

The [result index](results/nasbench201/README.md) lists raw traces, audits, hashes,
software details, test logs, and the rejected timing artifact.
