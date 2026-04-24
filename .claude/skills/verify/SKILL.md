---
name: verify
description: Smoke-test dl-gpumon after edits. Runs ruff, black --check, import sanity, and `--help` invocation. Use in place of a pytest suite (there isn't one).
---

# Verify

This repo has no pytest suite. Use this skill as the "did I break anything obvious" check after edits to `dl_gpumon.py`.

Run these four checks **in order**, stopping at the first failure and reporting what failed:

1. **Lint** — `ruff check .`
2. **Format** — `black --check .`
3. **Import sanity** — `python -c "import dl_gpumon"` (catches syntax errors and import-time exceptions that lint can't)
4. **CLI sanity** — `dl-gpumon --help` (catches regressions in the `argparse` setup and `main()` entry)

If a GPU is available and the user asks for a deeper check, additionally run `timeout 3 dl-gpumon -i 1 --no-dcgm` and report whether it started cleanly without a traceback. Do not run this unprompted — it requires an NVIDIA driver and terminal control.

Report a concise pass/fail line per step. On failure, quote the exact error and stop — do not try to fix it unless asked.
