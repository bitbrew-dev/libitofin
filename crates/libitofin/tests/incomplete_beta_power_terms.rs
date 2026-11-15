//! #1468: the incomplete beta prefactor keeps its precision for very large
//! `a`, `b`, where the `log_gamma` form cancelled to garbage. Oracle: mpmath
//! at 60 digits (`fixtures/ibeta_power_terms/generate.py`).

use libitofin::math::distributions::binomial::CumulativeBinomialDistribution;
use libitofin::types::Real;

/// Per-`n` tolerance of the Binomial rows: the continued fraction cancels near
/// the saddle at these sizes (a documented deferral in `math/beta.rs`).
fn binomial_tol(n: u64) -> Real {
    match n {
        1_000_000_000_000 => 5e-10,
        100_000_000_000_000 => 1e-8,
        _ => 5e-7,
    }
}

fn rows(kind: &str) -> Vec<Vec<Real>> {
    include_str!("fixtures/ibeta_power_terms/cases.csv")
        .lines()
        .filter_map(|line| line.strip_prefix(kind))
        .map(|line| line.split(',').map(|f| f.parse().unwrap()).collect())
        .collect()
}

fn assert_rel(got: Real, expected: Real, tol: Real, context: &str) {
    let err = ((got - expected) / expected).abs();
    assert!(
        got.is_finite() && err <= tol,
        "{context}: got {got}, expected {expected}, rel err {err:e}"
    );
}

#[test]
fn binomial_cdf_at_huge_n_matches_mpmath() {
    let rows = rows("binomial,");
    assert_eq!(rows.len(), 20);
    for row in rows {
        let (n, p, k, expected) = (row[0] as u64, row[1], row[2] as u64, row[3]);
        let got = CumulativeBinomialDistribution::new(p, n).unwrap().cdf(k);
        let context = format!("binomial n={n} p={p} k={k}");
        assert_rel(got, expected, binomial_tol(n), &context);
    }
}
