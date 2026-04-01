"""
stochastic-physics-simulations
================================
Python package for NVT Monte Carlo simulation of complex fluid systems.

Modules
-------
mcqhs
    Main MCMC engine — Metropolis-Hastings loop with Numba JIT compilation.
potentials
    Pair interaction models: Mie(50,49), Wigner-Kirkwood, Yoon-Scheraga,
    Lennard-Jones, and associated utility functions.
analysis
    Statistical post-processing: block averaging, autocorrelation,
    effective sample size, g(r) analysis, virial equation of state,
    and plotting utilities.

Usage
-----
    from potentials import mie_wks, mie_ys, xmin_poly
    from analysis   import load_uav, production_stats, plot_rdf
    import mcqhs                 # run full simulation via mcqhs.main()

References
----------
Jover et al., J. Chem. Phys. 137, 144505 (2012)
Wigner, Phys. Rev. 40, 749 (1932)
Kirkwood, Phys. Rev. 44, 31 (1933)
Yoon & Scheraga, J. Chem. Phys. 88, 3923 (1988)
Serna & Gil-Villegas, Mol. Phys. (2016)
"""

from .potentials import (
    mie_classical,
    mie_wks,
    mie_ys,
    lennard_jones,
    xmin_poly,
    cwk,
    du_dr,
    laplacian_mie,
    evaluate_all,
)

from .analysis import (
    load_uav,
    load_rdf,
    load_config,
    load_all,
    detect_burnin,
    production_stats,
    autocorrelation,
    integrated_autocorrelation_time,
    effective_sample_size,
    block_error,
    coordination_number,
    rdf_peaks,
    compressibility_virial,
    Z_qhs_theory,
    U_qhs_theory,
    plot_energy_convergence,
    plot_rdf,
    plot_Z_vs_rho,
    plot_collision_params,
    plot_internal_energy,
    summary_report,
)

__version__ = "1.0"
__author__  = "Ana Flores"
