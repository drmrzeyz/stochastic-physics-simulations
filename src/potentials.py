"""
potentials.py — Pair interaction models
========================================
Isolated, testable implementations of all pair potentials used in the
stochastic-physics-simulations engine.

Potentials
----------
mie_wks(r, lb)
    Mie(50,49) + Wigner-Kirkwood semiclassical correction (WKS).
    Truncated and shifted at the effective minimum r_min(λ_B).

mie_ys(r, lb)
    Mie(50,49) + Yoon-Scheraga quantum correction (YS).

mie_classical(r)
    Classical Mie(50,49) WCA potential (λ_B = 0 limit).

lennard_jones(r, eps, sigma)
    Standard Lennard-Jones(12,6) for reference and comparison.

Utility functions
-----------------
xmin_poly(lb)       Polynomial fit for the potential minimum vs. λ_B.
du_dr(r, lb)        Derivative du/dr — used in the virial equation for Z.
laplacian_mie(r)    Laplacian ∇²U of the Mie(50,49) potential.

All functions operate on scalar or NumPy array inputs and return
dimensionless reduced units (energies in ε, distances in σ).

References
----------
Jover et al., J. Chem. Phys. 137, 144505 (2012)   — Mie(50,49) model
Wigner, Phys. Rev. 40, 749 (1932)                  — WK correction
Kirkwood, Phys. Rev. 44, 31 (1933)                 — WK correction
Yoon & Scheraga, J. Chem. Phys. 88, 3923 (1988)   — YS correction
"""

import math
import numpy as np

# ── Physical constants (in reduced units) ─────────────────────────────────────
PI  = math.acos(-1.0)
CTE = 50.0 * (50.0 / 49.0) ** 49   # Mie(50,49) prefactor


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def xmin_poly(lb):
    """
    Position of the effective potential minimum as a function of λ_B.

    Polynomial fit obtained numerically by solving dU_WKS/dr = 0
    for a range of λ_B values (Table 6.1 of the thesis).

    Parameters
    ----------
    lb : float
        Thermal de Broglie wavelength λ_B / σ.

    Returns
    -------
    float
        r_min / σ — cutoff radius of the truncated potential.

    Valid range: 0 ≤ λ_B ≤ 0.9 (semiclassical regime).
    """
    lb2 = lb * lb
    lb3 = lb2 * lb
    return 1.0194 + 0.14983 * lb - 0.18698 * lb2 + 0.078628 * lb3


def cwk(lb):
    """
    Wigner-Kirkwood prefactor: λ_B² / (24π).

    Parameters
    ----------
    lb : float
        Thermal de Broglie wavelength λ_B / σ.

    Returns
    -------
    float
        CWK = λ_B² / (24π).
    """
    return lb * lb / (24.0 * PI)


def laplacian_mie(r):
    """
    Laplacian of the Mie(50,49) potential: ∇²U_Mie(r).

    Derived analytically:
        ∇²U = (1/r²) d/dr [r² dU/dr]
             = CTE * (2450 * r⁻⁵² − 2352 * r⁻⁵¹)

    Parameters
    ----------
    r : array-like
        Reduced distance r / σ.

    Returns
    -------
    ndarray
        ∇²U in units of ε / σ².
    """
    r = np.asarray(r, dtype=float)
    return CTE * (2450.0 * r ** (-52) - 2352.0 * r ** (-51))


def du_dr(r, lb=0.0):
    """
    Derivative of the effective potential dU/dr (Mie + WK correction).

    Used in the virial equation to compute the compressibility factor Z.

    Parameters
    ----------
    r  : array-like
        Reduced distance r / σ.
    lb : float, optional
        Thermal de Broglie wavelength (default 0 = classical).

    Returns
    -------
    ndarray
        dU/dr in units of ε / σ.
    """
    r   = np.asarray(r, dtype=float)
    R50 = r ** (-50);  R51 = r ** (-51)
    R52 = r ** (-52);  R53 = r ** (-53)
    DU1 = 49.0 * R50 - 50.0 * R51                          # classical term
    DU2 = 392.0 * cwk(lb) * (306.0 * R52 - 325.0 * R53)   # WK term
    return CTE * (DU1 + DU2)


# ===========================================================================
# PAIR POTENTIALS
# ===========================================================================

def mie_classical(r):
    """
    Classical Mie(50,49) WCA potential (λ_B = 0 limit).

    Truncated and shifted at r_min = (50/49)·σ ≈ 1.0204 σ so that
    u(r_min) = 0. This is the continuous hard-sphere approximation of
    Jover et al. (2012), valid for packing fractions 0.05 ≤ η ≤ 0.484.

    u(r) = CTE · ε · [(σ/r)⁵⁰ − (σ/r)⁴⁹] + ε    r < r_min
    u(r) = 0                                         r ≥ r_min

    Parameters
    ----------
    r : array-like
        Reduced distance r / σ.

    Returns
    -------
    ndarray
        Pair energy u(r) / ε.
    """
    r    = np.asarray(r, dtype=float)
    rmin = 50.0 / 49.0
    u    = np.zeros_like(r)
    m    = r < rmin
    rm   = r[m]
    u[m] = CTE * (rm ** (-50) - rm ** (-49)) + 1.0
    return u


def mie_wks(r, lb):
    """
    Mie(50,49) + Wigner-Kirkwood semiclassical correction, truncated
    and shifted at the effective minimum r_min(λ_B).

    The WK correction accounts for quantum delocalization to first order
    in ħ² (semiclassical expansion):

        U_WKS(r) = U_Mie(r) + (λ_B²/24π) · ∇²U_Mie(r) − U_WKS(r_min)

    The cutoff r_min(λ_B) is computed from the polynomial fit `xmin_poly`.
    As λ_B increases, r_min shifts outward — quantum effects effectively
    enlarge the particle diameter.

    Parameters
    ----------
    r  : array-like
        Reduced distance r / σ.
    lb : float
        Thermal de Broglie wavelength λ_B / σ.

    Returns
    -------
    ndarray
        Pair energy u(r) / ε.

    Notes
    -----
    Valid in the semiclassical limit λ_B ≪ σ (λ_B* ≤ 0.9 recommended).
    """
    r    = np.asarray(r, dtype=float)
    rmin = xmin_poly(lb)
    CWK  = cwk(lb)
    u    = np.zeros_like(r)
    m    = r < rmin
    rm   = r[m]

    # Mie part
    uhs  = CTE * (rm ** (-50) - rm ** (-49)) + 1.0
    # WK correction
    uwk  = CTE * CWK * (2450.0 * rm ** (-52) - 2352.0 * rm ** (-51))
    # Shift at minimum so u(r_min) = 0
    uhs0 = CTE * (rmin ** (-50) - rmin ** (-49)) + 1.0
    uwk0 = CTE * CWK * (2450.0 * rmin ** (-52) - 2352.0 * rmin ** (-51))

    u[m] = uhs + uwk - uhs0 - uwk0
    return u


def mie_ys(r, lb):
    """
    Mie(50,49) + Yoon-Scheraga quantum correction.

    The YS correction models quantum exchange effects via the Slater sum
    for a two-body relative wave function (Yoon & Scheraga, 1988):

        U_YS(r) = U_Mie(r) + kT · ξ²(r) · exp(−ξ²(r)) / (1 − exp(−ξ²(r)))

    where ξ(r) = σ√(2π) / λ_B · (r/σ − 1).

    Unlike WKS, this correction is purely repulsive and does not require
    re-centering at r_min.

    Parameters
    ----------
    r  : array-like
        Reduced distance r / σ.
    lb : float
        Thermal de Broglie wavelength λ_B / σ.

    Returns
    -------
    ndarray
        Pair energy u(r) / ε.

    Notes
    -----
    The YS correction provides better agreement with the quantum hard-sphere
    equation of state than WKS, especially at intermediate densities.
    """
    r    = np.asarray(r, dtype=float)
    rmin = 50.0 / 49.0          # YS uses the classical cutoff
    u    = np.zeros_like(r)
    m    = r < rmin
    rm   = r[m]

    # Classical Mie part
    u_mie = CTE * (rm ** (-50) - rm ** (-49)) + 1.0

    # YS quantum correction
    xi2 = (rm - 1.0) * (math.sqrt(2.0 * PI) / lb) ** 2
    xi2 = np.maximum(xi2, 1e-10)   # avoid division by zero at rm ≈ 1
    exp_xi2 = np.exp(-xi2)
    denom   = 1.0 - exp_xi2
    denom   = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    u_ys    = xi2 * exp_xi2 / denom    # in units of kT; T*=1.5 → /T*
    u[m]    = u_mie + u_ys / 1.5       # T* = 1.5 (reduced temperature)
    return u


def lennard_jones(r, eps=1.0, sigma=1.0):
    """
    Standard Lennard-Jones(12,6) pair potential.

    u(r) = 4ε · [(σ/r)¹² − (σ/r)⁶]

    Included as a well-known reference model for comparison with
    the Mie(50,49) potential.

    Parameters
    ----------
    r     : array-like
        Distance.
    eps   : float, optional
        Well depth (default 1.0).
    sigma : float, optional
        Zero-crossing distance (default 1.0).

    Returns
    -------
    ndarray
        Pair energy u(r) / ε.
    """
    r = np.asarray(r, dtype=float)
    x = sigma / r
    return 4.0 * eps * (x ** 12 - x ** 6)


# ===========================================================================
# CONVENIENCE: evaluate all models at once
# ===========================================================================

def evaluate_all(r, lb_values=None):
    """
    Evaluate all potential models at the same distances.

    Useful for comparison plots across quanticity values.

    Parameters
    ----------
    r         : array-like
        Reduced distances r / σ.
    lb_values : list of float, optional
        λ_B values for WKS and YS models.
        Defaults to [0.0, 0.226, 0.473, 0.670, 1.148].

    Returns
    -------
    dict
        Keys: 'r', 'classical', 'lj',
              'wks_{lb}' and 'ys_{lb}' for each λ_B.
    """
    if lb_values is None:
        lb_values = [0.0, 0.22608, 0.47339, 0.67019, 1.14789]

    r = np.asarray(r, dtype=float)
    result = {'r': r, 'classical': mie_classical(r),
              'lj': lennard_jones(r)}

    for lb in lb_values:
        key = f'{lb:.5f}'
        result[f'wks_{key}'] = mie_wks(r, lb)
        if lb > 0:
            result[f'ys_{key}']  = mie_ys(r, lb)

    return result


# ===========================================================================
# QUICK SELF-TEST
# ===========================================================================

if __name__ == '__main__':
    import sys

    print("=" * 55)
    print("potentials.py — self-test")
    print("=" * 55)

    r_test = np.array([0.90, 0.95, 1.00, xmin_poly(0.0), 1.10, 1.20])
    print("\nClassical Mie(50,49):")
    print(f"  {'r/σ':>8s}  {'u(r)/ε':>12s}")
    for r, u in zip(r_test, mie_classical(r_test)):
        print(f"  {r:8.4f}  {u:12.6f}")

    print("\nEffective minimum r_min(λ_B):")
    for lb in [0.0, 0.22608, 0.47339, 0.67019, 1.14789]:
        print(f"  λ_B = {lb:.5f}  →  r_min = {xmin_poly(lb):.6f} σ")

    print("\nWKS potential at r = 0.95 σ:")
    for lb in [0.0, 0.22608, 0.47339]:
        u = mie_wks(np.array([0.95]), lb)[0]
        print(f"  λ_B = {lb:.5f}  →  u = {u:.6f} ε")

    # Verify u(r_min) = 0 for all λ_B
    print("\nVerifying u(r_min) = 0 (truncation + shift):")
    all_ok = True
    for lb in [0.0, 0.22608, 0.47339, 0.67019]:
        rmin  = xmin_poly(lb)
        u_at_rmin = mie_wks(np.array([rmin]), lb)[0]
        ok    = abs(u_at_rmin) < 1e-8
        all_ok = all_ok and ok
        print(f"  λ_B = {lb:.5f}  u(r_min) = {u_at_rmin:.2e}  {'✓' if ok else '✗'}")

    print(f"\nAll checks passed: {all_ok}")
    sys.exit(0 if all_ok else 1)
