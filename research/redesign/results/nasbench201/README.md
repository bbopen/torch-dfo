# NASBench201 result files

Read the [study report](../../nasbench201-results.md) and [protocol](../../nasbench201-protocol.md) first.
The comparison has separate search-quality and execution-speed results.

| File | Contents |
| --- | --- |
| [analysis.json](analysis.json) | All method groups, failed counts, quality distributions, paired bootstrap intervals, and timing summaries |
| [quality-full.summary.json](quality-full.summary.json) | All 900 attempted searches and 720 final selected test scores |
| [quality-full.jsonl.gz](quality-full.jsonl.gz) | Every quality-run observation, including partial work from failed runs |
| [device-corrected-full.json.gz](device-corrected-full.json.gz) | Accepted timing artifact, including traces, graph counts, and failed method runs |
| [device-initial.json.gz](device-initial.json.gz) | Rejected compiled timing labels from the first panel; retain for audit |
| [device-initial.log](device-initial.log) | Recompilation-limit warning from the rejected panel |
| [device-corrected.log](device-corrected.log) | Corrected timing process log |
| [data-manifest.json](data-manifest.json) | Official archive locations, sizes, hashes, schema, and coverage |
| [upstream-parity.json](upstream-parity.json) | Exhaustive parity with the official EvoXBench evaluator |
| [evaluator-probes.json](evaluator-probes.json) | Degenerate, ceiling, repeated-trial shortcut, and random-reference probes |
| [quality-independent-audit.json](quality-independent-audit.json) | Independent audit of 723,600 quality observations and every final selection |
| [device-initial-independent-audit.json](device-initial-independent-audit.json) | Value audit for the rejected initial timing panel |
| [device-independent-audit.json](device-independent-audit.json) | Independent audit of 120,000 timing observations and compiled graph reuse |
| [audit_quality.py](audit_quality.py) | Independent quality-artifact checker used for this study |
| [audit_device.py](audit_device.py) | Independent corrected timing-artifact checker used for this study |
| [focused-tests.log](focused-tests.log) | Twelve focused benchmark tests passed |
| [regression.log](regression.log) | Complete Spark regression: 1,164 passed, 15 skipped, one expected failure |
| [manifest.json](manifest.json) | Artifact hashes, measured source commits, runtime, and execution order |
| [format-equivalence.json](format-equivalence.json) | AST equivalence after a formatting-only change to the measured device runner |

The quality panel used commit `c898ec5`. The corrected timing panel used commit
`22ad9f4bcd22c84dce260f0c60eb3dc5b1255e6d`. The current device runner differs only
in line wrapping; its Python syntax tree is unchanged. The [initial protocol](protocol-initial.md)
retains the quality panel's exact protocol bytes.

Both panels retain a `partial` status because stock EvoX CMA-ES failed on the tested
PyTorch build. The failed method receives no score or timing rank. A completed
comparison does not mean that every attempted method worked.

The full database remains external. Download it with the acquisition script and
check its hashes before reproducing the study. Decompress the trace files with
`gzip -dk` to inspect or rerun the independent artifact audits.
