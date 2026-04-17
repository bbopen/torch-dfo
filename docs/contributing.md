# Contributing

See [CONTRIBUTING.md](https://github.com/bbopen/torch-dfo/blob/master/CONTRIBUTING.md) in the repository root for the full development guide.

Key constraints:

- **No NumPy at runtime.** `src/torch_dfo/` must remain pure PyTorch. NumPy is allowed in `[dev]` / `[benchmarks]` extras, tests, and benchmark scripts only. A static AST guard in the test suite enforces this.
- **GPU-first.** All hot-path operations must be expressible as PyTorch tensor ops that run on CUDA, MPS, and CPU without branching on device type.
- **`torch.compile` compatibility.** Avoid Python-level branches on tensor values in `ask()` / `tell()`; use `torch.where` instead.
