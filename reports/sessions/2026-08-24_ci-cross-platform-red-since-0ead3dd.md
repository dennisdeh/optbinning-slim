# The macOS and Windows CI failures, diagnosed from the logs and fixed

## Ask

Get CI on `master` green. It had been failing on every push since `0ead3dd`.

## What was actually wrong

Two unrelated defects, one per platform, both introduced by `0ead3dd` and both
invisible on Linux — which is why five commits were merged on top of them.

**macOS — `test_weights_unvalidated_values`.** `_check_X_Y_weights` validated only
the *number* of scenario weights. `cp.py` makes them integral for CP-SAT with
`sw = 10 ** np.abs(np.log10(np.min(w)))` and then `np.int64(w[s] * sw)`. A negative
weight makes `sw` `NaN`; a zero makes it `inf`. **Casting a non-finite float to an
integer is undefined**, so what happened next was a property of the CPU: on x86-64
every weight became `INT64_MIN` and OR-Tools raised `TypeError: append():
incompatible function arguments`, while on macOS arm64 the same call *solved* and
returned splits. The test asserted that *something* was raised, so it passed on
Linux and Windows and failed on macOS. Same class as the MDLP `argsort` defect: an
expectation pinned on undefined behaviour.

**Windows — `test_documented_values.py`, all three tests.** It read the package
sources with `Path.read_text()` and no `encoding=`. That decodes with the locale
codec, which is cp1252 on the GitHub Windows runners, and several docstrings carry
typographic quotes (`Supported trends are “auto”` in `binning.py`, `U+201C`/`U+201D`).
Every one of the three died with `UnicodeDecodeError: 'charmap' codec can't decode
byte 0x9d` before asserting anything.

## What changed

| | |
|---|---|
| `optbinning/binning/uncertainty/binning_scenarios.py` | `_check_X_Y_weights` now validates the weight *values*: numeric dtype, finite, and strictly positive. Both `weights` docstrings updated to match. |
| `tests/test_binning_scenarios_edge_cases.py` | `test_weights_unvalidated_values` → `test_weights_values_are_validated`, now matching the error messages rather than accepting any exception, and covering the single-zero case. |
| `tests/test_documented_values.py` | Three `read_text()` calls given `encoding="utf-8"`. |
| `CHANGELOG.md`, `README.md` | A "Fixed — the 2026-08-24 cross-platform CI failures" section; README figures re-measured. |
| `CLAUDE.md` | New rule: a green local run is not a green CI, and check the run for the commit you are merging *onto*. |
| `reports/DECISIONS.md` | Two entries: why the weights are validated rather than left to OR-Tools, and why the sources are read as UTF-8 explicitly. |
| `reports/OPEN_ITEMS.md` | The part of the weight-scaling defect that is *not* fixed. |

**Strictly positive, not merely non-negative.** Zero is the `inf` case, and its scale
factor corrupts *every* weight rather than just its own: measured, `weights=[0., 1.]`
handed the model `[-9223372036854775808, -9223372036854775808]`. A zero weight is not
a way to drop a scenario, and the error message says to omit the scenario instead.

## Verification

Measured 2026-08-24, Python 3.13.15, on the rebased branch:

- **`pytest`** — **1110 passed, 0 failed**, 211 s. Warnings fell 26 → 23: the
  `RuntimeWarning: invalid value encountered in cast` at `cp.py:193` is gone, because
  bad weights no longer reach the model.
- **`flake8 --select=E9,F63,F7,F82`** — 0.
- **Both fixes demonstrated red first.**
  - Weights: with `binning_scenarios.py` reverted to master's version, the new test
    fails with the OR-Tools `TypeError` and the `cp.py:193` cast warning. Restored, it
    passes.
  - Encoding: the Windows failure was **reproduced on Linux** under
    `LC_ALL=C PYTHONUTF8=0`, which makes the locale codec ASCII. Master's file gives
    the identical failure on the identical character (`UnicodeDecodeError` on byte
    `0xe2`, the lead byte of the same `U+201C` whose trailing `0x9d` broke cp1252);
    the fixed file passes 10/10 under that same locale.
- The four failing CI job logs from run 23 were read directly through the Actions
  API. Failure counts: macOS 3.13 and 3.14 — 1 failed, 1109 passed; Windows 3.13 and
  3.14 — 3 failed, 1107 passed. Both ubuntu cells and the `package` job were green
  throughout.

## Left open

- **The weight scaling itself** still overflows `int64` for valid, strictly positive
  weights whose ratio exceeds about 1e18. Filed in `OPEN_ITEMS.md` with the measured
  threshold. Not fixed here because the repair changes the CP model's numerics and
  needs its own before/after measurement of the objective.
- CI had not been re-run against these fixes at the time of writing; the branch was
  pushed and the run started.
- The `package` job was never implicated, so the release automation added earlier the
  same day needed no change — but the `pypi` job is gated on the full matrix, so no
  release could have published while these two defects stood.
