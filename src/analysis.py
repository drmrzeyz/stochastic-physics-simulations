"""
analysis.py — Statistical post-processing and visualization
=============================================================
Functions for loading simulation output files, computing derived
observables, and generating publication-quality figures.

Sections
--------
I.   Loading output files
II.  Convergence diagnostics
III. Statistical estimators (block averaging, ACF, ESS)
IV.  Structural observables (g(r), coordination number)
V.   Equation of state (Z via virial equation)
VI.  Plotting utilities

All plotting functions return (fig, axes) so the caller can further
customize or save the figure.
"""

import math
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import integrate

warnings.filterwarnings("ignore")

PI = math.acos(-1.0)


# ===========================================================================
# I. LOADING OUTPUT FILES
# ===========================================================================

def load_uav(path="uav.dat"):
    """
    Load the time-series output file produced by the MCMC engine.

    File format (space-separated, one row per block):
        NMOV  UHS  UWK  UAV  std1  CV  std2

    Parameters
    ----------
    path : str
        Path to uav.dat.

    Returns
    -------
    pd.DataFrame
        Columns: NMOV, UHS, UWK, UAV, std1, CV, std2.
    """
    df = pd.read_csv(path, sep=r"\s+", header=0)
    df.columns = ["NMOV", "UHS", "UWK", "UAV", "std1", "CV", "std2"]
    return df


def load_rdf(path="rdf.dat"):
    """
    Load the radial distribution function file.

    File format (space-separated, no header):
        r   g(r)

    Parameters
    ----------
    path : str
        Path to rdf.dat.

    Returns
    -------
    pd.DataFrame
        Columns: r, gr.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["r", "gr"])
    return df


def load_config(path="mc.new"):
    """
    Load a particle configuration file (mc.new or mc.old).

    File format (space-separated, one row per particle):
        RX  RY  RZ

    Parameters
    ----------
    path : str
        Path to the configuration file.

    Returns
    -------
    np.ndarray
        Shape (N, 3) — particle positions in box units.
    """
    return np.loadtxt(path)


def load_all(output_dir="../data/outputs"):
    """
    Load all standard output files from a simulation run.

    Parameters
    ----------
    output_dir : str
        Directory containing uav.dat and rdf.dat.

    Returns
    -------
    dict
        Keys: 'uav' (DataFrame), 'rdf' (DataFrame).
        Missing files are silently skipped.
    """
    result = {}
    for name, loader, fname in [
        ("uav", load_uav,    "uav.dat"),
        ("rdf", load_rdf,    "rdf.dat"),
    ]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            result[name] = loader(fpath)
        else:
            warnings.warn(f"{fpath} not found — skipping.")
    return result


# ===========================================================================
# II. CONVERGENCE DIAGNOSTICS
# ===========================================================================

def detect_burnin(uav_series, window=5, threshold_frac=0.05):
    """
    Detect the end of the burn-in phase from the energy time series.

    Method: computes the running mean of the block energy series, then
    finds the first block where the absolute slope of the running mean
    (in a sliding window) drops below `threshold_frac` of its maximum.

    Parameters
    ----------
    uav_series    : array-like
        Energy per particle per block.
    window        : int
        Number of blocks in the sliding window for slope estimation.
    threshold_frac : float
        Fraction of max slope below which the chain is considered equilibrated.

    Returns
    -------
    int
        Index of the first production block (0-based).
    """
    u     = np.asarray(uav_series)
    n     = len(u)
    rmean = np.cumsum(u) / np.arange(1, n + 1)
    slopes = []
    for k in range(window, n):
        y = rmean[k - window : k]
        slope = np.polyfit(np.arange(window), y, 1)[0]
        slopes.append(abs(slope))
    slopes = np.array(slopes)
    thresh = slopes.max() * threshold_frac
    idx    = np.argmax(slopes < thresh)
    return int(idx + window) if slopes[idx] < thresh else n // 3


def production_stats(df_uav, eq_idx=None):
    """
    Compute production-phase statistics from the block time series.

    Parameters
    ----------
    df_uav : pd.DataFrame
        Output of `load_uav`.
    eq_idx : int, optional
        Index of the first production block. If None, auto-detected.

    Returns
    -------
    dict
        Keys: eq_idx, n_prod, U_mean, U_stderr, U_block_std,
              CV_mean, CV_stderr, accept_mean.
    """
    if eq_idx is None:
        eq_idx = detect_burnin(df_uav["UAV"].values)

    prod = df_uav.iloc[eq_idx:]
    n    = len(prod)
    return dict(
        eq_idx      = eq_idx,
        n_prod      = n,
        U_mean      = prod["UAV"].mean(),
        U_stderr    = prod["std1"].mean() / math.sqrt(n),
        U_block_std = prod["UAV"].std(),
        CV_mean     = prod["CV"].mean(),
        CV_stderr   = prod["CV"].std() / math.sqrt(n),
        accept_mean = None,   # acceptance rate not stored in uav.dat
    )


# ===========================================================================
# III. STATISTICAL ESTIMATORS
# ===========================================================================

def autocorrelation(series, max_lag=None):
    """
    Normalized autocorrelation function (ACF) of a 1D series.

    Parameters
    ----------
    series  : array-like
    max_lag : int, optional
        Maximum lag to compute. Defaults to len(series) // 2.

    Returns
    -------
    lags : np.ndarray
    acf  : np.ndarray
        Normalized ACF (acf[0] = 1).
    """
    x   = np.asarray(series) - np.mean(series)
    n   = len(x)
    if max_lag is None:
        max_lag = n // 2
    full = np.correlate(x, x, mode="full")
    acf  = full[n - 1 :]
    acf  = acf[:max_lag] / acf[0]
    return np.arange(max_lag), acf


def integrated_autocorrelation_time(acf):
    """
    Integrated autocorrelation time τ_int from the ACF array.

    τ_int = 0.5 + Σ_{k=1}^{K} acf(k)   (sum while acf(k) > 0)

    Parameters
    ----------
    acf : array-like
        Normalized ACF starting at lag 0.

    Returns
    -------
    float
        τ_int in units of blocks.
    """
    return 0.5 + sum(a for a in acf[1:] if a > 0)


def effective_sample_size(series):
    """
    Effective sample size (ESS) corrected for autocorrelation.

    ESS = N / (2 · τ_int)

    Parameters
    ----------
    series : array-like
        Time series (e.g., block energy averages).

    Returns
    -------
    float
        ESS.
    """
    _, acf  = autocorrelation(series)
    tau_int = integrated_autocorrelation_time(acf)
    return len(series) / (2.0 * max(tau_int, 0.5))


def block_error(series):
    """
    Standard error of the mean estimated via block averaging.

    σ_err = std(block_means) / sqrt(n_blocks)

    Parameters
    ----------
    series : array-like
        Time series of block means.

    Returns
    -------
    float
        Estimated standard error.
    """
    bm = np.asarray(series)
    return bm.std() / math.sqrt(len(bm))


# ===========================================================================
# IV. STRUCTURAL OBSERVABLES
# ===========================================================================

def coordination_number(r, gr, rho, r_max=None):
    """
    Running coordination number n(r) = integral of 4π·ρ·g(r)·r² dr.

    Parameters
    ----------
    r     : array-like
        Radial distances r / σ.
    gr    : array-like
        Radial distribution function g(r).
    rho   : float
        Reduced number density ρ*.
    r_max : float, optional
        Upper integration limit. If None, integrates over all r.

    Returns
    -------
    float
        Coordination number n(r_max).
    """
    r  = np.asarray(r);  gr = np.asarray(gr)
    if r_max is not None:
        mask = r <= r_max
        r, gr = r[mask], gr[mask]
    integrand = 4.0 * PI * rho * gr * r ** 2
    return float(integrate.trapezoid(integrand, r))


def rdf_peaks(r, gr):
    """
    Locate the first peak and first minimum of g(r).

    Parameters
    ----------
    r  : array-like
    gr : array-like

    Returns
    -------
    dict
        Keys: r_peak, g_peak, r_min1, g_min1.
    """
    r  = np.asarray(r);  gr = np.asarray(gr)
    peak_mask = (r > 0.90) & (r < 1.40)
    pi        = np.argmax(gr[peak_mask])
    r_peak    = r[peak_mask][pi]
    g_peak    = gr[peak_mask][pi]
    min_mask  = (r > r_peak) & (r < r_peak + 0.8)
    if min_mask.sum() > 0:
        mi    = np.argmin(gr[min_mask])
        r_min1 = r[min_mask][mi]
        g_min1 = gr[min_mask][mi]
    else:
        r_min1, g_min1 = r_peak + 0.4, 0.5
    return dict(r_peak=r_peak, g_peak=g_peak,
                r_min1=r_min1, g_min1=g_min1)


# ===========================================================================
# V. EQUATION OF STATE
# ===========================================================================

def compressibility_virial(r, gr, rho, lb=0.0):
    """
    Compressibility factor Z via the virial equation of state.

    Z = 1 + (2π·ρ/3) · ∫ g(r) · r · (dU/dr) · r² dr

    Separates repulsive (dU/dr > 0) and attractive (dU/dr < 0)
    contributions to give ν_R, ν_A, ⟨s⟩, ⟨l⟩, and Z.

    Parameters
    ----------
    r   : array-like
        Radial distances r / σ.
    gr  : array-like
        Radial distribution function g(r).
    rho : float
        Reduced number density ρ*.
    lb  : float
        Thermal de Broglie wavelength λ_B / σ (default 0 = classical).

    Returns
    -------
    dict
        Keys: Z, freqr (ν_R), freqa (ν_A), s_rep (⟨s⟩), s_att (⟨l⟩).
    """
    from potentials import du_dr as _du_dr, xmin_poly

    r  = np.asarray(r, dtype=float)
    gr = np.asarray(gr, dtype=float)

    rmin = xmin_poly(lb)
    mask = (r >= 0.88) & (r <= rmin)
    rv, grv = r[mask], gr[mask]

    DU   = _du_dr(rv, lb)
    DELR = np.gradient(rv)

    rep  = DU > 0
    att  = DU < 0

    def _sum(cond, power):
        return (grv[cond] * DU[cond] * rv[cond] ** power * DELR[cond]).sum()

    fr   = _sum(rep, 2);  scol_r = _sum(rep, 3)
    fa   = abs(_sum(att, 2));  scol_a = _sum(att, 3)

    s_rep  = scol_r / fr   if fr  != 0 else 0.0
    s_att  = scol_a / fa   if fa  != 0 else 0.0
    freqr  = -24.0 * fr * rho
    freqa  =  24.0 * fa * rho
    Z      = 1.0 + (2.0 * PI / 3.0) * (s_rep * freqr - s_att * freqa)

    return dict(Z=Z, freqr=freqr, freqa=freqa,
                s_rep=s_rep, s_att=s_att)


def Z_qhs_theory(rho, lb):
    """
    Compressibility factor from the quantum hard-sphere equation of state.

    Uses the Carnahan-Starling equation with an effective packing fraction
    that accounts for quantum corrections (Serna & Gil-Villegas, 2016).

    Parameters
    ----------
    rho : float or array-like
        Reduced number density ρ*.
    lb  : float
        Thermal de Broglie wavelength λ_B / σ.

    Returns
    -------
    float or ndarray
        Z = PV/NkT.
    """
    rho = np.asarray(rho, dtype=float)
    lb2 = lb * lb
    et  = rho * PI / 6.0;  et2 = et ** 2
    d1  = 1.6593854484;  d2 = -1.0927115150;  d3 = -1.1188233921
    etq  = (1.0 + d1*lb)*et + (d2*lb + d3*lb2)*et2
    etq2 = etq ** 2;  etq3 = etq2 * etq
    detq3 = (1.0 - etq) ** 3
    return (1.0 + etq + etq2 - etq3) / detq3


def U_qhs_theory(rho, lb):
    """
    Internal energy per particle from the quantum hard-sphere EOS.

    Derived analytically from A_QHS via the thermodynamic relation
    U = A − T·(∂A/∂T) (Eq. 6.18 of the thesis).

    Parameters
    ----------
    rho : float or array-like
    lb  : float

    Returns
    -------
    float or ndarray
        ⟨U⟩ / NkT.
    """
    rho = np.asarray(rho, dtype=float)
    lb2 = lb * lb
    et  = rho * PI / 6.0;  et2 = et ** 2
    d1  = 1.6593854484;  d2 = -1.0927115150;  d3 = -1.1188233921
    etq = (1.0 + d1*lb)*et + (d2*lb + d3*lb2)*et2
    detq3 = (1.0 - etq) ** 3
    dxle  = d1*et + (d2 + 2.0*d3*lb)*et2
    TEMP  = 0.5 / (lb2 * PI)
    return (((2.0 - etq) * lb * et) / detq3) * dxle * math.sqrt(TEMP) \
           * 0.5 / math.sqrt(2.0 * PI)


# ===========================================================================
# VI. PLOTTING UTILITIES
# ===========================================================================

LB_VALUES = [0.0, 0.22608, 0.47339, 0.67019, 1.14789]
LB_LABELS = [
    r"$\lambda_B^* = 0.0$  (classical)",
    r"$\lambda_B^* = 0.226$",
    r"$\lambda_B^* = 0.473$",
    r"$\lambda_B^* = 0.670$",
    r"$\lambda_B^* = 1.148$",
]
COLORS  = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]
MARKERS = ["o", "s", "^", "D", "v"]


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def plot_energy_convergence(df_uav, eq_idx=None, title=None):
    """
    Three-panel energy convergence plot.

    Panels: (1) energy components, (2) running mean, (3) block std.

    Parameters
    ----------
    df_uav  : pd.DataFrame
    eq_idx  : int, optional
    title   : str, optional

    Returns
    -------
    fig, axes
    """
    _apply_style()
    if eq_idx is None:
        eq_idx = detect_burnin(df_uav["UAV"].values)

    steps = df_uav["NMOV"].values
    uav   = df_uav["UAV"].values
    s1    = df_uav["std1"].values
    rmean = np.cumsum(uav) / np.arange(1, len(uav) + 1)
    eq_step = steps[eq_idx]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(steps, uav,   color="steelblue", lw=1.8, label="U/N per block")
    axes[0].plot(steps, df_uav["UHS"], color="darkorange", lw=1.2,
                 ls="--", alpha=0.8, label="U_HS/N")
    axes[0].plot(steps, df_uav["UWK"], color="seagreen",   lw=1.2,
                 ls=":",  alpha=0.8, label="U_WK/N")
    axes[0].fill_between(steps, uav - s1, uav + s1, alpha=0.15, color="steelblue")
    axes[0].axvline(eq_step, color="tomato", ls="--", lw=1.5)
    axes[0].axvspan(steps[0], eq_step, alpha=0.06, color="tomato")
    axes[0].set_ylabel("U / N  (ε)")
    axes[0].set_title("Energy Components")
    axes[0].legend(fontsize=9, ncol=3)

    axes[1].plot(steps, rmean, color="navy", lw=2.0, label="Running mean")
    axes[1].axvline(eq_step, color="tomato", ls="--", lw=1.5,
                    label=f"Burn-in end (step {eq_step:,})")
    prod_mean = rmean[eq_idx:]
    axes[1].axhspan(prod_mean.mean() - prod_mean.std(),
                    prod_mean.mean() + prod_mean.std(),
                    alpha=0.12, color="seagreen", label="Production ±1σ")
    axes[1].set_ylabel("Running mean U/N")
    axes[1].set_title("Running Mean Convergence")
    axes[1].legend(fontsize=9)

    axes[2].semilogy(steps, s1, color="darkorchid", lw=1.6)
    axes[2].axvline(eq_step, color="tomato", ls="--", lw=1.5)
    axes[2].set_ylabel("Block std σ₁  (ε)")
    axes[2].set_xlabel("MCMC step")
    axes[2].set_title("Sampling Error per Block")

    for ax in axes:
        ax.axvspan(steps[0], eq_step, alpha=0.04, color="tomato")

    if title:
        fig.suptitle(title, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_rdf(df_rdf, rho=None, title=None):
    """
    Two-panel radial distribution function plot.

    Left: g(r) with annotations. Right: g(r) − 1 (shell structure).

    Parameters
    ----------
    df_rdf : pd.DataFrame
    rho    : float, optional  (for coordination number)
    title  : str, optional

    Returns
    -------
    fig, axes
    """
    _apply_style()
    r  = df_rdf["r"].values
    gr = df_rdf["gr"].values
    pk = rdf_peaks(r, gr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(r, gr, color="steelblue", lw=2.0)
    axes[0].axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.6,
                    label="Ideal gas g(r) = 1")
    axes[0].axvline(1.0, color="lightcoral", lw=1.0, ls=":", alpha=0.7,
                    label="r = σ")
    axes[0].fill_between(r, 0, gr, where=(r < 1.0),
                         alpha=0.12, color="tomato", label="Excluded core")
    axes[0].annotate(
        f"Peak  r={pk['r_peak']:.3f}σ\ng={pk['g_peak']:.2f}",
        xy=(pk["r_peak"], pk["g_peak"]),
        xytext=(pk["r_peak"] + 0.25, pk["g_peak"] * 0.85),
        fontsize=9, color="navy",
        arrowprops=dict(arrowstyle="->", color="navy"),
    )
    axes[0].set_xlabel("r / σ");  axes[0].set_ylabel("g(r)")
    axes[0].set_title("Radial Distribution Function", fontweight="bold")
    axes[0].set_xlim(0.75, r.max());  axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=9)

    dev = gr - 1.0
    axes[1].fill_between(r, dev, 0, where=(dev >= 0),
                         alpha=0.25, color="steelblue", label="g(r) > 1")
    axes[1].fill_between(r, dev, 0, where=(dev < 0),
                         alpha=0.25, color="tomato",   label="g(r) < 1")
    axes[1].plot(r, dev, color="navy", lw=1.8)
    axes[1].axhline(0, color="gray", lw=0.7, ls="--")
    axes[1].set_xlabel("r / σ");  axes[1].set_ylabel("g(r) − 1")
    axes[1].set_title("Deviation from Ideal Gas", fontweight="bold")
    axes[1].set_xlim(0.75, r.max())
    axes[1].legend(fontsize=9)

    if title:
        fig.suptitle(title, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig, axes


def plot_Z_vs_rho(all_data, lb_values=None, show_theory=True):
    """
    Compressibility factor Z vs. density for multiple λ_B values.

    Parameters
    ----------
    all_data   : list of dict
        Each dict must have keys 'rho' and 'Z' (lists).
    lb_values  : list of float, optional
    show_theory : bool

    Returns
    -------
    fig, ax
    """
    _apply_style()
    if lb_values is None:
        lb_values = LB_VALUES[:len(all_data)]

    fig, ax = plt.subplots(figsize=(9, 6))
    rho_th  = np.linspace(0.04, 0.88, 200)

    for data, lb, label, color, marker in zip(
            all_data, lb_values, LB_LABELS, COLORS, MARKERS):
        rho = np.array(data["rho"])
        Z   = np.array(data["Z"])
        ax.plot(rho, Z, color=color, marker=marker,
                lw=0, ms=7, label=label)
        if show_theory:
            ax.plot(rho_th, Z_qhs_theory(rho_th, lb),
                    color=color, lw=1.4, ls="--", alpha=0.65)

    ax.axhline(1.0, color="gray", lw=0.8, ls=":", alpha=0.6,
               label="Ideal gas  Z = 1")
    ax.set_xlabel(r"$\rho^*$");  ax.set_ylabel("Z = PV/NkT")
    ax.set_title("Compressibility Factor\n"
                 "(markers = MC,  dashed = theory)", fontweight="bold")
    handles = [Line2D([0],[0], color=c, marker=m, lw=0, ms=7, label=lb)
               for c, m, lb in zip(COLORS, MARKERS, LB_LABELS[:len(all_data)])]
    handles += [Line2D([0],[0], color="gray", lw=1.5, ls="--",
                        alpha=0.7, label="Theory (QHS)"),
                Line2D([0],[0], color="gray", lw=0.8, ls=":",
                        label="Ideal gas")]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.set_ylim(-25, 25)
    plt.tight_layout()
    return fig, ax


def plot_collision_params(all_data, lb_values=None):
    """
    ⟨s⟩ and ⟨l⟩ vs. density for multiple λ_B values.

    Parameters
    ----------
    all_data  : list of dict  (each needs 'rho', 's', 'l')
    lb_values : list of float, optional

    Returns
    -------
    fig, axes
    """
    _apply_style()
    if lb_values is None:
        lb_values = LB_VALUES[:len(all_data)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for data, lb, label, color, marker in zip(
            all_data, lb_values, LB_LABELS, COLORS, MARKERS):
        rho = np.array(data["rho"])
        axes[0].plot(rho, data["s"], color=color, marker=marker,
                     lw=1.8, ms=6, label=label)
        axes[1].plot(rho, data["l"], color=color, marker=marker,
                     lw=1.8, ms=6, label=label)

    axes[0].set_xlabel(r"$\rho^*$");  axes[0].set_ylabel(r"$\langle s \rangle/\sigma$")
    axes[0].set_title(r"Mean Repulsive Diameter $\langle s \rangle$", fontweight="bold")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel(r"$\rho^*$");  axes[1].set_ylabel(r"$\langle l \rangle/\sigma$")
    axes[1].set_title(r"Mean Attractive Range $\langle l \rangle$", fontweight="bold")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig, axes


def plot_internal_energy(rho_arr, U_mc_dict, U_qhs_dict=None, title=None):
    """
    Internal energy ⟨U⟩/NkT vs. density.

    Parameters
    ----------
    rho_arr    : array-like
    U_mc_dict  : dict  {lb: list_of_U_values}
    U_qhs_dict : dict, optional  {lb: list_of_U_theory} (None = skip theory)
    title      : str, optional

    Returns
    -------
    fig, ax
    """
    _apply_style()
    rho = np.asarray(rho_arr)
    fig, ax = plt.subplots(figsize=(10, 6))

    for lb, color, marker in zip(
            list(U_mc_dict.keys()), COLORS[1:], MARKERS[1:]):
        label = LB_LABELS[LB_VALUES.index(lb)] if lb in LB_VALUES else f"λ_B={lb}"
        ax.plot(rho, U_mc_dict[lb], color=color, marker=marker,
                lw=1.6, ms=5, label=label)
        if U_qhs_dict and lb in U_qhs_dict:
            u_th = U_qhs_dict[lb]
            valid = [(r, u) for r, u in zip(rho, u_th) if u is not None]
            if valid:
                rv, uv = zip(*valid)
                ax.plot(rv, uv, color=color, lw=1.4, ls="--", alpha=0.65)

    ax.set_xlabel(r"$\rho^*$");  ax.set_ylabel(r"$\langle U \rangle / NkT$")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    return fig, ax


def summary_report(df_uav, df_rdf=None, rho=None, lb=0.0):
    """
    Print a concise summary of all key observables.

    Parameters
    ----------
    df_uav : pd.DataFrame
    df_rdf : pd.DataFrame, optional
    rho    : float, optional  (required for Z and coordination number)
    lb     : float, optional
    """
    stats  = production_stats(df_uav)
    _, acf = autocorrelation(df_uav["UAV"].iloc[stats["eq_idx"]:].values)
    tau    = integrated_autocorrelation_time(acf)
    ess    = stats["n_prod"] / (2.0 * max(tau, 0.5))

    print("=" * 56)
    print("  SIMULATION SUMMARY REPORT")
    print("=" * 56)
    print(f"  Burn-in ends at block : {stats['eq_idx']}")
    print(f"  Production blocks     : {stats['n_prod']}")
    print(f"  Autocorr. time τ_int  : {tau:.2f}  blocks")
    print(f"  Effective sample size : {ess:.0f}  ({100*ess/stats['n_prod']:.1f}%)")
    print()
    print(f"  Mean energy  U/N      : {stats['U_mean']:.5f} ± {stats['U_stderr']:.5f}  ε")
    print(f"  Heat capacity C_V/N   : {stats['CV_mean']:.4f} ± {stats['CV_stderr']:.4f}  k_B")

    if df_rdf is not None:
        pk = rdf_peaks(df_rdf["r"].values, df_rdf["gr"].values)
        print()
        print(f"  g(r) contact peak     : {pk['g_peak']:.4f}  at r = {pk['r_peak']:.4f} σ")
        print(f"  First minimum         : r = {pk['r_min1']:.4f} σ")
        if rho is not None:
            n1 = coordination_number(df_rdf["r"].values, df_rdf["gr"].values,
                                     rho, r_max=pk["r_min1"])
            print(f"  Coordination number   : {n1:.2f}")
            Z_dict = compressibility_virial(
                df_rdf["r"].values, df_rdf["gr"].values, rho, lb)
            print(f"  Compressibility Z     : {Z_dict['Z']:.5f}")
    print("=" * 56)


# ===========================================================================
# QUICK SELF-TEST
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("analysis.py — self-test")

    # Synthetic data
    np.random.seed(0)
    n = 20
    steps = np.arange(1, n + 1) * 50000
    uav   = -0.17 + 0.05 * np.exp(-np.arange(n) / 3) + 0.004 * np.random.randn(n)
    df    = pd.DataFrame({
        "NMOV": steps, "UHS": uav * 0.73, "UWK": uav * 0.27,
        "UAV": uav, "std1": np.abs(0.003 * np.random.randn(n)) + 0.001,
        "CV": 1.23 + 0.05 * np.random.randn(n),
        "std2": np.abs(0.04 * np.random.randn(n)) + 0.005,
    })

    eq = detect_burnin(df["UAV"].values)
    print(f"  Burn-in detected at block {eq}")

    stats = production_stats(df, eq)
    print(f"  U/N = {stats['U_mean']:.5f} ± {stats['U_stderr']:.5f}")
    print(f"  C_V/N = {stats['CV_mean']:.4f}")

    _, acf = autocorrelation(df["UAV"].values)
    tau    = integrated_autocorrelation_time(acf)
    ess    = effective_sample_size(df["UAV"].values)
    print(f"  τ_int = {tau:.2f},  ESS = {ess:.0f}")

    # Z theory
    Z30 = Z_qhs_theory(0.30, 0.22608)
    print(f"  Z_QHS(ρ=0.30, λ=0.226) = {Z30:.5f}  (expected ≈ 2.507)")

    print("All checks passed ✓")
    sys.exit(0)
