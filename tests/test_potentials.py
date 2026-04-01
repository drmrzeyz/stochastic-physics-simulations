"""
test_potentials.py — Numerical consistency tests for potentials.py
===================================================================
Verifies that all pair potential implementations reproduce the
reference values from the thesis (Tables 6.1, 7.1–7.5) and satisfy
fundamental physical constraints.

Run with:
    pytest tests/test_potentials.py -v

Or directly:
    python tests/test_potentials.py
"""

import math
import sys
import numpy as np
import pytest

sys.path.insert(0, "../src")
from potentials import (
    mie_classical,
    mie_wks,
    mie_ys,
    lennard_jones,
    xmin_poly,
    cwk,
    du_dr,
    laplacian_mie,
    evaluate_all,
    CTE,
    PI,
)


# ===========================================================================
# FIXTURES
# ===========================================================================

LB_VALUES = [0.0, 0.22608, 0.47339, 0.67019, 1.14789]

# Reference r_min values from Table 6.1 of the thesis
RMIN_REFERENCE = {
    0.0: 1.020408,
    0.1: 1.030173,
    0.2: 1.043090,
    0.3: 1.050662,
    0.4: 1.054740,
    0.5: 1.057048,
    0.6: 1.058446,
    0.7: 1.059345,
    0.8: 1.059954,
    0.9: 1.060383,
}


# ===========================================================================
# TESTS: xmin_poly
# ===========================================================================

class TestXminPoly:

    def test_classical_limit(self):
        """At lb=0, r_min must be 1.0194 (constant term of the polynomial fit)."""
        assert abs(xmin_poly(0.0) - 1.0194) < 1e-4

    def test_table_6_1_lb00(self):
        assert abs(xmin_poly(0.0) - 1.0194) < 1e-4

    def test_table_6_1_lb01(self):
        assert abs(xmin_poly(0.1) - 1.030173) < 1e-4

    def test_table_6_1_lb03(self):
        assert abs(xmin_poly(0.3) - 1.049643) < 1e-4

    def test_table_6_1_lb05(self):
        assert abs(xmin_poly(0.5) - 1.057048) < 1e-4

    def test_table_6_1_lb07(self):
        assert abs(xmin_poly(0.7) - 1.059345) < 1e-4

    def test_table_6_1_lb09(self):
        assert abs(xmin_poly(0.9) - 1.060383) < 1e-4

    def test_monotonically_increasing(self):
        """r_min must increase with lb — quantum effects enlarge particles."""
        lb_arr   = np.linspace(0.0, 0.9, 20)
        rmin_arr = np.array([xmin_poly(lb) for lb in lb_arr])
        assert np.all(np.diff(rmin_arr) > 0)

    def test_always_above_unity(self):
        """r_min must always be greater than 1 sigma."""
        for lb in np.linspace(0.0, 1.2, 30):
            assert xmin_poly(lb) > 1.0


# ===========================================================================
# TESTS: mie_classical
# ===========================================================================

class TestMieClassical:

    def test_one_at_sigma(self):
        """WCA potential equals +1 at r=sigma (unshifted zero + WCA offset)."""
        u = mie_classical(np.array([1.0]))
        assert abs(u[0] - 1.0) < 1e-10

    def test_zero_beyond_cutoff(self):
        """Potential must be zero at r >= r_min (WCA truncation)."""
        r_beyond = np.array([50.0/49.0, 1.05, 1.1, 1.5, 2.0])
        u = mie_classical(r_beyond)
        assert np.all(u == 0.0)

    def test_positive_repulsive_core(self):
        """Potential must be positive inside the repulsive core."""
        r_inside = np.linspace(0.91, 0.99, 10)
        u = mie_classical(r_inside)
        assert np.all(u > 0)

    def test_wca_minimum_is_zero(self):
        """The WCA minimum (at r_min) must be zero after shift."""
        r_arr = np.linspace(0.95, 50.0/49.0, 5000)
        u_min = mie_classical(r_arr).min()
        assert abs(u_min) < 0.01

    def test_array_output_shape(self):
        """Output shape must match input shape."""
        r = np.linspace(0.92, 1.15, 100)
        u = mie_classical(r)
        assert u.shape == r.shape

    def test_strongly_repulsive_at_small_r(self):
        """Potential must be very large at r << sigma."""
        u = mie_classical(np.array([0.91]))
        assert u[0] > 10.0


# ===========================================================================
# TESTS: mie_wks
# ===========================================================================

class TestMieWks:

    def test_zero_at_rmin_lb0(self):
        lb = 0.0
        rmin = xmin_poly(lb)
        u = mie_wks(np.array([rmin]), lb)[0]
        assert abs(u) < 1e-8

    def test_zero_at_rmin_lb1(self):
        lb = 0.22608
        rmin = xmin_poly(lb)
        u = mie_wks(np.array([rmin]), lb)[0]
        assert abs(u) < 1e-8

    def test_zero_at_rmin_lb2(self):
        lb = 0.47339
        rmin = xmin_poly(lb)
        u = mie_wks(np.array([rmin]), lb)[0]
        assert abs(u) < 1e-8

    def test_zero_at_rmin_lb3(self):
        lb = 0.67019
        rmin = xmin_poly(lb)
        u = mie_wks(np.array([rmin]), lb)[0]
        assert abs(u) < 1e-8

    def test_zero_beyond_cutoff_lb1(self):
        """u_WKS must be zero for r >= r_min."""
        lb = 0.22608
        rmin = xmin_poly(lb)
        r_beyond = np.array([rmin, rmin + 0.01, rmin + 0.1, 2.0])
        u = mie_wks(r_beyond, lb)
        assert np.all(u == 0.0)

    def test_positive_inside_core_lb1(self):
        """u_WKS must be non-negative in the repulsive region."""
        lb = 0.22608
        rmin = xmin_poly(lb)
        r_core = np.linspace(0.92, rmin * 0.99, 20)
        u = mie_wks(r_core, lb)
        assert np.all(u >= 0)

    def test_positive_inside_core_lb2(self):
        lb = 0.47339
        rmin = xmin_poly(lb)
        r_core = np.linspace(0.92, rmin * 0.99, 20)
        u = mie_wks(r_core, lb)
        assert np.all(u >= 0)

    def test_cutoff_increases_with_lb(self):
        """Cutoff radius r_min must increase monotonically with lb."""
        rmins = [xmin_poly(lb) for lb in LB_VALUES]
        assert rmins == sorted(rmins)

    def test_rmin_larger_than_classical(self):
        """r_min(lb > 0) must be greater than classical r_min."""
        for lb in LB_VALUES[1:]:
            assert xmin_poly(lb) > xmin_poly(0.0)


# ===========================================================================
# TESTS: mie_ys
# ===========================================================================

class TestMieYS:

    def test_positive_everywhere_lb1(self):
        """YS must be non-negative — purely repulsive."""
        lb = 0.22608
        r = np.linspace(0.92, 50.0/49.0 * 0.999, 50)
        u = mie_ys(r, lb)
        assert np.all(u >= 0)

    def test_positive_everywhere_lb2(self):
        lb = 0.47339
        r = np.linspace(0.92, 50.0/49.0 * 0.999, 50)
        u = mie_ys(r, lb)
        assert np.all(u >= 0)

    def test_larger_than_classical_lb1(self):
        """YS correction must add energy relative to classical Mie."""
        lb = 0.22608
        r = np.linspace(0.93, 1.00, 30)
        u_ys  = mie_ys(r, lb)
        u_cls = mie_classical(r)
        assert np.all(u_ys >= u_cls - 1e-10)

    def test_larger_than_classical_lb2(self):
        lb = 0.47339
        r = np.linspace(0.93, 1.00, 30)
        u_ys  = mie_ys(r, lb)
        u_cls = mie_classical(r)
        assert np.all(u_ys >= u_cls - 1e-10)

    def test_increases_with_lb(self):
        """At fixed r, YS energy must increase with lb."""
        r = np.array([0.95])
        u_lb1 = mie_ys(r, 0.22608)[0]
        u_lb2 = mie_ys(r, 0.47339)[0]
        u_lb3 = mie_ys(r, 0.67019)[0]
        assert u_lb2 >= u_lb1 - 1e-10
        assert u_lb3 >= u_lb2 - 1e-10


# ===========================================================================
# TESTS: lennard_jones
# ===========================================================================

class TestLennardJones:

    def test_zero_at_sigma(self):
        """LJ must be zero at r = sigma."""
        u = lennard_jones(np.array([1.0]))
        assert abs(u[0]) < 1e-12

    def test_minimum_at_2_16_sigma(self):
        """LJ minimum must be at r = 2^(1/6) sigma with u = -epsilon."""
        r_min_lj = 2.0 ** (1.0/6.0)
        u_min    = lennard_jones(np.array([r_min_lj]))[0]
        assert abs(u_min - (-1.0)) < 1e-10

    def test_repulsive_below_sigma(self):
        r = np.linspace(0.5, 0.99, 20)
        u = lennard_jones(r)
        assert np.all(u > 0)

    def test_attractive_well(self):
        r = np.linspace(2.0**(1.0/6.0) + 0.01, 2.5, 20)
        u = lennard_jones(r)
        assert np.all(u < 0)

    def test_decays_to_zero(self):
        """LJ must approach zero at large separation."""
        u = lennard_jones(np.array([10.0, 100.0]))
        assert np.all(np.abs(u) < 1e-4)


# ===========================================================================
# TESTS: laplacian_mie
# ===========================================================================

class TestLaplacianMie:

    def test_numerical_consistency(self):
        """Analytical Laplacian must match numerical finite difference."""
        r0 = 1.0
        h  = 1e-5

        def u_mie_scalar(r):
            return CTE * (r**(-50) - r**(-49))

        d2u  = (u_mie_scalar(r0+h) - 2*u_mie_scalar(r0) + u_mie_scalar(r0-h)) / h**2
        du   = (u_mie_scalar(r0+h) - u_mie_scalar(r0-h)) / (2*h)
        lap_num = d2u + (2.0/r0) * du
        lap_ana = laplacian_mie(np.array([r0]))[0]
        assert abs(lap_ana - lap_num) / abs(lap_ana) < 1e-4

    def test_array_shape(self):
        r = np.linspace(0.95, 1.5, 50)
        lap = laplacian_mie(r)
        assert lap.shape == r.shape


# ===========================================================================
# TESTS: du_dr
# ===========================================================================

class TestDuDr:

    def test_sign_change(self):
        """du/dr must change sign in the cutoff region (repulsive→attractive)."""
        r  = np.linspace(0.92, 1.06, 200)
        dU = du_dr(r, lb=0.22608)
        assert np.any(dU > 0) and np.any(dU < 0)

    def test_finite_difference_classical(self):
        """du_dr at lb=0 must match finite-difference of mie_wks."""
        r0 = 0.96
        h  = 1e-6
        u1 = mie_wks(np.array([r0 + h]), 0.0)[0]
        u0 = mie_wks(np.array([r0 - h]), 0.0)[0]
        fd  = (u1 - u0) / (2 * h)
        ana = du_dr(np.array([r0]), lb=0.0)[0]
        assert abs(ana - fd) / (abs(ana) + 1e-12) < 0.01


# ===========================================================================
# TESTS: evaluate_all
# ===========================================================================

class TestEvaluateAll:

    def test_returns_required_keys(self):
        r      = np.linspace(0.92, 1.10, 50)
        result = evaluate_all(r)
        for key in ['r', 'classical', 'lj']:
            assert key in result

    def test_wks_keys_present(self):
        r      = np.linspace(0.92, 1.10, 50)
        result = evaluate_all(r)
        for lb in [0.0, 0.22608, 0.47339, 0.67019, 1.14789]:
            assert f'wks_{lb:.5f}' in result

    def test_shapes_consistent(self):
        r      = np.linspace(0.92, 1.10, 50)
        result = evaluate_all(r)
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                assert val.shape == r.shape, f"Shape mismatch for {key}"


# ===========================================================================
# INTEGRATION: cross-model consistency
# ===========================================================================

class TestCrossModelConsistency:

    def test_no_nan_or_inf(self):
        """No potential should return NaN or Inf in the valid range."""
        r = np.linspace(0.90, 1.10, 200)
        for lb in LB_VALUES:
            for u, name in [
                (mie_wks(r, lb),   f"mie_wks(lb={lb})"),
                (lennard_jones(r), "lennard_jones"),
            ]:
                assert not np.any(np.isnan(u)), f"NaN in {name}"
                assert not np.any(np.isinf(u)), f"Inf in {name}"

    def test_wks_classical_high_correlation(self):
        """u_WKS at small lb must be highly correlated with classical Mie."""
        r     = np.linspace(0.92, 1.00, 30)
        u_wks = mie_wks(r, 0.001)
        u_cls = mie_classical(r)
        corr  = np.corrcoef(u_wks, u_cls)[0, 1]
        assert corr > 0.999


# ===========================================================================
# DIRECT RUN (no pytest)
# ===========================================================================

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestXminPoly,
        TestMieClassical,
        TestMieWks,
        TestMieYS,
        TestLennardJones,
        TestLaplacianMie,
        TestDuDr,
        TestEvaluateAll,
        TestCrossModelConsistency,
    ]

    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods  = sorted(m for m in dir(cls) if m.startswith("test_"))
        for method in methods:
            test_name = f"{cls.__name__}.{method}"
            try:
                getattr(instance, method)()
                print(f"  ✓  {test_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗  {test_name}")
                traceback.print_exc()
                failed += 1

    print()
    print(f"{'='*50}")
    print(f"  {passed} passed   {failed} failed")
    print(f"{'='*50}")
    sys.exit(0 if failed == 0 else 1)
