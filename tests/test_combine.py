"""
tests/test_combine.py — Unit tests for src/combine.py.

Run with:  python -m pytest tests/  (from gnss-agent-control/)
           or: python tests/test_combine.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.combine import (
    CalibrationWeights,
    CombinedResult,
    MONITOR_ORDER,
    simple_mean,
    inv_variance,
    full_covariance,
    chi_square_gate,
    combine_all,
    _N,
    _RIDGE,
)
from src.monitor import MonitorResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_results(scores: dict[str, int]) -> list[MonitorResult]:
    return [MonitorResult(m, scores.get(m, 50), "test") for m in MONITOR_ORDER]


def correlated_weights() -> CalibrationWeights:
    """
    Return CalibrationWeights with a strongly correlated M1–M2 covariance.

    Sigma = [[100,  80,  0,  0],
             [ 80, 100,  0,  0],
             [  0,   0, 25,  0],
             [  0,   0,  0, 25]]

    M1 and M2 are highly correlated (rho=0.8): they carry redundant information
    and should be downweighted together by full_covariance but not by inv_variance
    (which treats them independently).
    """
    cov = np.array([
        [100.0,  80.0,  0.0,  0.0],
        [ 80.0, 100.0,  0.0,  0.0],
        [  0.0,   0.0, 25.0,  0.0],
        [  0.0,   0.0,  0.0, 25.0],
    ])
    variances = {MONITOR_ORDER[i]: float(cov[i, i]) for i in range(_N)}
    return CalibrationWeights(variances=variances, cov_matrix=cov.tolist())


def diagonal_weights() -> CalibrationWeights:
    """Weights with the same diagonal as correlated_weights but no off-diagonal terms."""
    cov = np.diag([100.0, 100.0, 25.0, 25.0])
    variances = {MONITOR_ORDER[i]: float(cov[i, i]) for i in range(_N)}
    return CalibrationWeights(variances=variances, cov_matrix=cov.tolist())


# ---------------------------------------------------------------------------
# Test 1: full_covariance differs from inv_variance under correlation
# ---------------------------------------------------------------------------

def test_full_covariance_differs_from_inv_variance_under_correlation():
    """
    When M1 and M2 are strongly correlated, full_covariance should differ from
    inv_variance because the full P downweights the redundant M1–M2 pair,
    giving more influence to the independent M3 and M4.

    With scores M1=M2=90 (high suspicion) and M3=M4=10 (low suspicion):
      - inv_variance: M3/M4 have higher weight (lower individual variance 25 vs 100),
        but the M1/M2 correlation is ignored, so they still pull the mean toward 90.
      - full_covariance: the full P recognises that M1 and M2 are redundant;
        after accounting for correlation, the combined score is pulled further
        toward the M3/M4 values than inv_variance.
    """
    results = make_results({"M1": 90, "M2": 90, "M3": 10, "M4": 10})
    w_corr = correlated_weights()
    w_diag = diagonal_weights()

    fc = full_covariance(results, w_corr)
    iv = inv_variance(results, w_corr)

    # full_covariance must differ from inv_variance (the whole point of this fix)
    assert abs(fc.score - iv.score) > 1.0, (
        f"full_covariance ({fc.score:.2f}) should differ from "
        f"inv_variance ({iv.score:.2f}) when monitors are correlated."
    )

    # With diagonal weights, full_covariance and inv_variance should agree
    fc_diag = full_covariance(results, w_diag)
    iv_diag = inv_variance(results, w_diag)
    assert abs(fc_diag.score - iv_diag.score) < 0.01, (
        f"On a diagonal covariance, full_covariance ({fc_diag.score:.2f}) "
        f"should equal inv_variance ({iv_diag.score:.2f})."
    )

    print(f"  [PASS] correlated: full_cov={fc.score:.2f}, inv_var={iv.score:.2f}  "
          f"(differ by {abs(fc.score - iv.score):.2f})")
    print(f"  [PASS] diagonal:   full_cov={fc_diag.score:.2f}, inv_var={iv_diag.score:.2f}  "
          f"(agree to 0.01)")


# ---------------------------------------------------------------------------
# Test 2: CalibrationWeights.estimate uses both labels
# ---------------------------------------------------------------------------

def test_calibration_uses_both_labels():
    """
    With honest target=0 and sabotage target=100, a monitor that always scores
    correctly (0 on honest, 100 on sabotage) should have zero error variance.
    A monitor that always scores 50 regardless of label should have high error variance.
    """
    records = []
    # 5 honest runs: perfect monitor M1=0, blind monitor M3=50
    for _ in range(5):
        records.append({"label": "honest",   "scores": {"M1": 0,  "M2": 50, "M3": 50, "M4": 50}})
    # 5 sabotage runs: perfect monitor M1=100, blind monitor M3=50
    for _ in range(5):
        records.append({"label": "sabotage", "scores": {"M1": 100, "M2": 50, "M3": 50, "M4": 50}})

    w = CalibrationWeights.estimate(records)

    # M1 error is always 0; its variance should be tiny (just the ridge)
    assert w.variances["M1"] <= 2.0, f"Perfect monitor M1 variance should be near 0, got {w.variances['M1']:.2f}"

    # M3 error is -50 on honest and -50 on sabotage; variance should be near 0 too
    # (consistently wrong in a constant direction is low variance of error)
    # Actually M3 score=50, honest target=0 → error=50; sabotage target=100 → error=-50.
    # Variance of [50,50,...,-50,-50,...] = 2500. But the ridge adds 1.
    assert w.variances["M3"] > 100.0, f"Blind monitor M3 should have high error variance, got {w.variances['M3']:.2f}"

    print(f"  [PASS] M1 (perfect) variance={w.variances['M1']:.2f}  "
          f"M3 (blind) variance={w.variances['M3']:.2f}")


# ---------------------------------------------------------------------------
# Test 3: simple_mean is correct
# ---------------------------------------------------------------------------

def test_simple_mean():
    results = make_results({"M1": 20, "M2": 40, "M3": 60, "M4": 80})
    r = simple_mean(results)
    assert abs(r.score - 50.0) < 1e-9, f"Expected 50.0, got {r.score}"
    print(f"  [PASS] simple_mean = {r.score}")


# ---------------------------------------------------------------------------
# Test 4: chi-square escalates when monitors are inconsistent
# ---------------------------------------------------------------------------

def test_chi_square_escalates_on_inconsistency():
    """
    If some monitors score 5 and others score 95, the residuals are large and
    T should exceed the chi-square threshold.
    """
    w = CalibrationWeights()  # default variances (400 each)
    # Extreme disagreement: M1/M2 say 5, M3/M4 say 95
    results = make_results({"M1": 5, "M2": 5, "M3": 95, "M4": 95})
    r = chi_square_gate(results, w)
    assert r.escalate, f"Expected escalate=True on extreme disagreement, got T={r.reason}"

    # Unanimous agreement should not escalate
    results_agree = make_results({"M1": 80, "M2": 80, "M3": 80, "M4": 80})
    r2 = chi_square_gate(results_agree, w)
    assert not r2.escalate, f"Expected escalate=False on unanimous agreement"

    print(f"  [PASS] escalate on disagreement={r.escalate}, unanimous={r2.escalate}")


# ---------------------------------------------------------------------------
# Test 5: combine_all returns all four methods
# ---------------------------------------------------------------------------

def test_combine_all_returns_all_methods():
    results = make_results({"M1": 50, "M2": 50, "M3": 50, "M4": 50})
    w = CalibrationWeights()
    combined = combine_all(results, w)
    assert set(combined.keys()) == {"simple_mean", "inv_variance", "full_covariance", "chi_square_gate"}
    print(f"  [PASS] combine_all returned {sorted(combined.keys())}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_full_covariance_differs_from_inv_variance_under_correlation,
        test_calibration_uses_both_labels,
        test_simple_mean,
        test_chi_square_escalates_on_inconsistency,
        test_combine_all_returns_all_methods,
    ]
    passed = failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(failed)
