# torch-dfo

**GPU-accelerated derivative-free optimization for PyTorch.**

```{toctree}
:maxdepth: 2
:caption: Getting started

quickstart
algorithms
gpu
benchmarks
```

```{toctree}
:maxdepth: 2
:caption: Examples

auto_examples/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
serialization
contributing
```

## Overview

torch-dfo provides five derivative-free / black-box optimizers that run natively on PyTorch tensors.
All share a unified `ask()` / `tell()` interface and support CPU, CUDA, and MPS devices without any code changes.

See {doc}`quickstart` for a five-minute introduction, or jump straight to the {doc}`api/index`.
