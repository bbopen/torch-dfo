# Foundation review and correction batch

Base commit: `e393a3449c8e86bac7859a0adc16f6d4d6204a78`.
Local branch: `review/foundation-20260919`.

This review inspected the core algorithms, state helpers, phased search, mixed spaces, optimizer wrapper, benchmark harness, examples, and package claims. It reproduced defects before correction. Separate reviewers checked the correction lanes. This is a bounded source review and tested correction batch, not proof that the library has no remaining defects.

## Corrected behavior

| Area | Reproduced failure | Correction |
| --- | --- | --- |
| SHADE checkpoints | A checkpoint before first `ask()` lost the pending warm-start population | Save and restore the pending population |
| CMA-ES restart | A nondefault initial sigma reverted to a hardcoded fraction | Preserve the configured fraction through restart and checkpoints |
| Bounds | Wrong-length tensor bounds failed late; infinite bounds produced NaNs | Reject invalid shape and nonfinite bounds at construction |
| Population sizes | CMA-ES population 1 and SHADE population 2 failed after construction | Enforce supported minimum sizes |
| Model wrapper budget | Budget 5 with population 4 evaluated 8 candidates | Stop before an incomplete generation; preserve state and unused residue |
| DLR budget | Branch populations 8, 12, 20 produced 20 candidates under a total cap of 14 | Select branches by cumulative population |
| Wrapper checkpoint reuse | Loading removed keys from the caller's mapping | Read from a local mapping copy |
| Phased budget | Budget 7 made 10 real objective calls | Reject oversized atomic batches and use the existing bounded polish path |
| Integer spaces | A singleton interval constructed successfully, then divided by zero in encoding | Reject singleton intervals, consistent with current Float handling |
| Closure contract | Two closure modes silently selected one | Reject ambiguous calls before evaluation |
| COCO success | A local minimum was used as a global optimum, yielding false solved results | Use COCO's official target flag; leave unavailable precision unknown |
| COCO identity | Requested instance 6 actually selected instance 71 | Select literal instances and record COCO's actual ID |
| pycma budget | A 25-call budget made 30 calls | Count every call and stop a final partial generation without updating CMA-ES |
| YAHPO reporting | Comparison depended on row order; repeated baseline cost was collapsed | Aggregate repeats and preserve each repeat's cost; reject nonfinite wins |
| Examples and claims | Wrapper and structured-space examples did not match the API; GPU/compile claims exceeded evidence | Run corrected examples and state the supported behavior and limits |

The benchmark extra now includes `cma`, which the default baseline needs. Documentation no longer presents historical memory ceilings as general optimizer recommendations.

## Verification

- Full local suite: **1,096 passed, 3 skipped**, 37 warnings, 52.33 seconds.
- Coverage: **92.19%**, above the configured 85% gate.
- Ruff passed for source, tests, changed benchmark harness, and changed example.
- Wheel and source distribution built successfully.
- Independent core checks covered minimal populations, malformed bounds, SHADE state before and after ask/tell, and nondefault CMA restart state.
- Independent composite checks covered cumulative branch selection, unused budget residues, checkpoint reuse, and exact next-candidate preservation.
- Independent live COCO checks covered budgets 1, 2, 3, 7, 25, 40, 49, 50 at dimensions 2 and 10.
- Revised model-wrapper, structured-search, and neural-network examples executed.

The first baseline suite ran 1,038 tests successfully, then its Python 3.14 coverage collector failed. Switching coverage to the supported `sysmon` core produced the clean final gate above. The failed collector run remains in the evidence.

Local environment: macOS arm64, Python 3.14.4, PyTorch 2.10.0. The three skipped tests concern optional dependencies. This result does not replace the repository's full supported-version CI matrix.

## Behavior comparison

An 18-case CPU comparison used COCO functions 1, 3, and 8, literal instance 6, dimensions 2 and 10, and budgets 7, 40, and 200.

The original code exceeded the cap in six cases. The corrected code exceeded it in none. Final quality improved in five cases, matched in eight, and worsened in five. These are single-seed diagnostics, not a superiority study. The stricter phase boundary can change the search trajectory even when both runs ultimately use the same number of calls.

One example is Rosenbrock at dimension 10 and budget 200: the old result was about -71.32, and the corrected result about 66.01. Lower is better. The fix establishes the budget contract; it does not establish an improved allocation policy. A redesigned scheduler must preserve the cap and test its quality separately.

## Remaining design concerns

- `SearchSpace.decode()` extracts scalar values and builds Python dictionaries. Structured search is not an entirely GPU-resident path.
- Categorical encoding can evaluate the same decoded design repeatedly. Deduplication must be explicit because repeated noisy evaluations can be meaningful.
- The polish implementation includes Rastrigin-specific smoothing assumptions. A generic claim needs ablations on unseen problem families.
- YAHPO still compares one torch-dfo seed with several random seeds. The README now labels that comparison descriptive.
- The Gymnasium benchmark uses a simple policy and does not establish robust control performance or GPU acceleration.
- Nonfinite fitness, failed evaluations, noisy replicates, and constraints need a clearer public result contract.
- The current tests do not certify global convergence, every state transition, every device/version combination, or any scientific application.

The current branch is a preserved foundation for the newly authorized redesign. No changes were pushed or published by this review.
