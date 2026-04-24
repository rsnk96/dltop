---
name: verify
description: Smoke-test cvtop after edits. Runs ruff, black --check, pytest, and `cvtop --help`. The "did I break anything obvious" check.
---

# Verify

Run these checks **in order**, stopping at the first failure and reporting what failed:

1. **Lint** — `ruff check .`
2. **Format** — `black --check .`
3. **Tests** — `pytest -q` (smoke tests in `tests/`; no GPU needed)
4. **CLI sanity** — `cvtop --help` (catches regressions in argparse / `main()` entry that import-only tests miss)

If a GPU is available and the user asks for a deeper check, additionally run `timeout 3 cvtop -i 1 --no-dcgm` and report whether it started cleanly without a traceback. Do not run this unprompted — it requires an NVIDIA driver and terminal control.

Report a concise pass/fail line per step. On failure, quote the exact error and stop — do not try to fix it unless asked.
