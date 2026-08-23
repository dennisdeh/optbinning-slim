# Open items

Things the code does that it should not. One entry per defect; remove the entry
in the commit that fixes it.

## Piecewise binning: WoE expectations drift beyond tolerance

*Last updated: 2026-08-23*

`tests/test_binning_piecewise.py::test_bounds_transform` and
`::test_bounds_fit_transform` fail on the first four transformed WoE values:

```
Index | Obtained           | Expected
0     | 3.9918103483894214 | 3.99180564 ± 4.0e-06
1     | 4.282455817895456  | 4.28245092 ± 4.3e-06
2     | 4.174079851403313  | 4.17407503 ± 4.2e-06
3     | -3.256559765566703 | -3.2565373 ± 3.3e-06
```

Max absolute difference 2.2e-05, max relative 6.9e-06, against a `rel=1e-6`
`approx` tolerance. Measured 2026-08-23 with ropwr 1.2.0, cvxpy 1.9.2,
scipy 1.18.1, numpy 2.5.2 on Python 3.12.

Not yet root-caused. The shape (a uniform small offset, correct sign and
magnitude everywhere) is consistent with solver/BLAS drift in the underlying
`ropwr` fit rather than a logic defect, but that has not been demonstrated —
do not close this on the assumption. The question to answer first is whether
the fitted objective is unchanged; if it is, the fix is to loosen the
tolerance, and if it is not, something in the piecewise path changed
behaviour. Deciding either way belongs in `DECISIONS.md`.

These two are the only known failures of the suite as of 2026-08-23
(2 failed, 173 passed).
