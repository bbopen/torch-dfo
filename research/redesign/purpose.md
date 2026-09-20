# Purpose and working direction

Owner clarification, 19 September 2026.

## The product

Torch-dfo is a maintained derivative-free optimization library built on PyTorch for engineering and research. It includes algorithms, configurable search, tested behavior, documentation, benchmarks, and practical applications.

The same repository will contain an autonomous research lab. The lab will propose changes, run experiments, check results, and apply useful improvements. Known problems provide reference evaluations. Engineering case studies demonstrate use. Exploratory applications test new ideas and expose missing capabilities.

The library and lab develop together. Application work should exercise the public library instead of accumulating separate private implementations. General improvements should become maintained library features. Optional application dependencies should not burden every library installation.

## Implementation sources

Use EvoTorch as a direct source of implementation code as well as design knowledge. Inspect how its algorithms work. Reuse or adapt suitable implementations, correct demonstrated weaknesses, and test the resulting behavior.

There is no requirement to rewrite a useful implementation independently. Record the upstream revision and changed behavior. Keep source attribution with adapted code and credit EvoTorch and its contributors in the README.

The same standard applies to our existing code. Preserve useful parts, replace weak parts, and let evidence decide. Our lineage does not require preserving every architectural choice.

## Motivation

Brett's website describes a goal of helping people live longer, more abundant lives. It identifies autonomous hardware design and the space economy as major interests. These statements support a broad engineering and research program. [Brett Bonner](https://brett-bonner.com/), read on 19 September 2026.

For this project, that suggests applications such as hardware and controller co-design, simulator-based engineering design, robust mission or resource planning, and biological model calibration. These are candidate research directions, not established capabilities or results.

Biology and control are current evaluation candidates. They do not define the limits of the library. The current two-adapter pilot is an experiment limit, not a permanent restriction on applications.

## How the lab works

Keep evaluators separate from the implementation under test. Freeze each experiment's data, objective, budget, and acceptance rule before tuning. Preserve failed trials and independent validation results.

Use published benchmarks to establish correctness and comparison quality. Use practical applications to test whether the interface and algorithms solve useful problems. Use exploratory demonstrations to investigate hypotheses that established benchmarks do not cover.

A successful application can justify a new operator, representation, constraint policy, execution method, or optimizer. Retain that capability when it has a clear interface, tests, documentation, and evidence that another user can reproduce.

## Current status

The foundation correction batch is preserved at `e446f68`. The redesign contracts and evaluation proposals exist. The new architecture and autonomous lab have not yet been implemented. Direct EvoTorch adaptation is now an explicit part of the implementation plan.
