# Security Policy

## Reporting a Vulnerability

Please do **not** file a public GitHub issue for security vulnerabilities.

Report them privately via GitHub's [private vulnerability reporting](https://github.com/bbopen/torch-dfo/security/advisories/new) feature.

Include:
- A description of the vulnerability and its potential impact.
- Steps to reproduce or a minimal proof-of-concept.
- Affected versions (if known).

You will receive a response within 7 days acknowledging the report. We aim to release a patch within 30 days of confirmation.

## Scope

torch-dfo is a numerical optimization library with no network access, no authentication logic, and no persistent storage of user data. The primary security surface is the `state_dict` / `load_state_dict` interface, which deserializes optimizer state from arbitrary dicts. **Do not call `load_state_dict` on untrusted input** — treat it the same as `torch.load` with `weights_only=False`.
