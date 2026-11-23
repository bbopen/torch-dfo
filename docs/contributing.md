# Contributing

See [CONTRIBUTING.md](https://github.com/bbopen/torch-dfo/blob/master/CONTRIBUTING.md) in the repository root for setup and PR guidelines.

Core repository invariants:

- **Pure PyTorch runtime.** `src/torch_dfo/` must not import NumPy, SciPy, Pandas, JAX, or other CPU-bound libraries (`tests/test_no_cpu_binding_imports.py`).
- **Self-documenting library code.** Keep `src/torch_dfo/` free of non-directive inline `#` comments (`tests/test_no_inline_comments.py`); use clear names and docstrings instead.
- **Formal math and boundary checks.** Run `formal/verify.sh` and `tests/test_formal_invariants.py` when editing optimizer update rules, budget accounting, or numerical sanitization.
- **Device parity.** Keep tensor hot paths portable across CPU, CUDA, and MPS (`float32` on MPS).
