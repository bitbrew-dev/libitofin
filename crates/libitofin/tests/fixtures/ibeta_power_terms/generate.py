"""Oracle for libitofin #1468 (incomplete beta power terms at large a, b).

Rows are computed with mpmath at 60 significant digits and rounded to 17;
inputs are binary64 values printed exactly.

- `lanczos,z,value`: Boost's `lanczos13m53::lanczos_sum_expG_scaled(z)`, the
  exact `Gamma(z) e^(z - 1/2) / (z + g - 1/2)^(z - 1/2)` at Boost's binary64 g.

Regenerate from the repository root:

    uv run --with mpmath==1.3.0 python -I \
        crates/libitofin/tests/fixtures/ibeta_power_terms/generate.py \
        > crates/libitofin/tests/fixtures/ibeta_power_terms/cases.csv
"""

import mpmath

mpmath.mp.dps = 60

LANCZOS_G = mpmath.mpf(6.024680040776729583740234375)
LANCZOS_Z = [1e-3, 0.25, 0.5, 1.0, 1.5, 2.0, 3.5, 10.0, 47.25, 1e3, 1e6, 5e11, 5e15, 1e20]


def lanczos_scaled(z):
    z = mpmath.mpf(z)
    gh = LANCZOS_G - mpmath.mpf(1) / 2
    return mpmath.exp(mpmath.loggamma(z) + z - mpmath.mpf(1) / 2 - (z - mpmath.mpf(1) / 2) * mpmath.log(z + gh))


def fmt(x):
    return mpmath.nstr(x, 17, min_fixed=-1, max_fixed=1)


def main():
    print(f"# mpmath {mpmath.__version__}, dps {mpmath.mp.dps}; kind,inputs...,value")
    for z in LANCZOS_Z:
        print(f"lanczos,{z!r},{fmt(lanczos_scaled(z))}")


if __name__ == "__main__":
    main()
