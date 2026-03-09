# Stochastic Physics Simulations
(work in progress...)

> Stochastic simulation engine for evaluating the stability of complex systems, using variance reduction techniques.

A high-performance **Markov Chain Monte Carlo (MCMC)** engine built in Python, designed to sample from complex probability distributions and estimate statistical observables in many-body systems. Originally developed for a Master's thesis in Physics, the codebase was ported from Fortran 77 to Python with **Numba JIT compilation**, achieving near-native performance while gaining full reproducibility and extensibility.

---

## Technical Highlights

- **MCMC sampling** via the Metropolis-Hastings algorithm on a high-dimensional continuous state space
- **Variance reduction** through block averaging and statistical error propagation across independent subsamples
- **Numerical performance engineering**: critical loops compiled with Numba `@njit`, replacing legacy Fortran without sacrificing speed
- **Statistical estimators**: mean, variance, heat capacity (from energy fluctuations), pair correlation functions, and compressibility
- **Legacy code modernization**: full translation of a Fortran 77 scientific codebase to idiomatic, documented Python

---

## How It Works

The engine simulates a system of **N interacting particles** in a periodic 3D box. At each step of the Markov chain:

1. A particle is selected at random
2. A trial displacement is proposed from a uniform distribution
3. The energy change ΔU is computed
4. The move is accepted with probability min(1, exp(−ΔU / T)) — the **Metropolis-Hastings criterion**
5. Statistical accumulators are updated regardless of acceptance

This generates a Markov chain whose stationary distribution is the target Boltzmann distribution — a canonical example of MCMC applied to a continuous, high-dimensional sample space.

**Variance reduction** is applied through two complementary strategies:

- **Block averaging**: the trajectory is divided into independent blocks; the standard deviation across block means estimates the true sampling error, correcting for autocorrelation in the chain
- **Incremental updates**: only the energy contribution of the displaced particle is recomputed at each step — O(N) instead of O(N²) — dramatically reducing estimator noise per unit of compute

---

## Repository Structure

```
stochastic-physics-simulations/
│
├── README.md
│
├── notebooks/
│   ├── 01_model_and_sampling.ipynb       # MCMC setup, proposal distribution, convergence
│   ├── 02_simulation_walkthrough.ipynb   # Step-by-step annotated simulation run
│   └── 03_results_analysis.ipynb         # Observables, variance analysis, pair correlations
│
├── src/
│   ├── __init__.py
│   ├── mcqhs.py                          # Main MCMC engine (Numba-accelerated)
│   ├── potentials.py                     # Pair interaction model (isolated and testable)
│   └── analysis.py                       # Statistical post-processing and plotting
│
├── original_fortran/
│   ├── mcqhs.f                           # Original Fortran 77 source
│   └── mc.inc                            # Shared variable declarations
│
├── data/
│   ├── inputs/
│   │   └── mc.in                         # Example simulation parameters
│   └── outputs/
│       ├── mc.dat                        # Main results per block
│       ├── uav.dat                       # Time series for convergence diagnostics
│       └── rdf.dat                       # Pair correlation function g(r)
│
├── tests/
│   └── test_potentials.py                # Numerical consistency tests vs. Fortran reference
│
└── requirements.txt                      # Python dependencies
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/drmrzeyz/stochastic-physics-simulations.git
cd stochastic-physics-simulations
```

```bash
pip install -r requirements.txt
```

### 2. Run a simulation

```bash
cd src
python mcqhs.py
```

> The first run triggers Numba JIT compilation (~20–30 s). All subsequent runs start immediately.

### 3. Explore the notebooks

```bash
jupyter lab notebooks/
```

Start with `01_model_and_sampling.ipynb` for a conceptual walkthrough of the MCMC setup, then move to the analysis notebook for visualizations.

---

## Key Parameters (`mc.in`)

| Parameter | Description |
|---|---|
| `N` | Number of particles (system size) |
| `RHO` | Reduced number density |
| `NMOVE` | Total number of MCMC steps |
| `NSUB` | Block size for variance estimation |
| `DISPL` | Maximum trial displacement — controls the acceptance rate |
| `NFDR` | Sampling frequency for the pair correlation function |
| `NRUN` | `0` = initialize from lattice, `1` = resume from previous state |

A typical target acceptance rate is 30–50%, tuned by adjusting `DISPL`.

---

## Outputs

| File | Contents |
|---|---|
| `mc.dat` | Block averages: energy, acceptance rate, heat capacity, compressibility factor |
| `uav.dat` | Full time series for convergence and mixing diagnostics |
| `rdf.dat` | Pair correlation function g(r) — structural fingerprint of the sampled configurations |
| `mc.new` | Final system state (use with `NRUN=1` to resume sampling) |

---

## Performance

The computational bottleneck is pairwise energy evaluation inside the Metropolis loop. Key optimizations applied:

- **Incremental updates**: only the moved particle's interactions are recomputed per step — O(N) per move instead of O(N²)
- **Numba `@njit`**: the inner loop compiles to machine code at first call, matching Fortran-level throughput
- **Scalar accumulators**: no intermediate array allocation inside the hot loop

On a modern laptop (single core), the engine processes ~10⁶ MCMC steps in under 2 minutes for N = 108 particles.

---

## Background

This engine was developed as part of a **Master's thesis in Physics**, where MCMC methods were used to study the statistical behavior of complex systems under uncertainty. The original Fortran 77 implementation was translated to Python to improve reproducibility, readability, and integration with modern data science workflows.

The core methods — Metropolis-Hastings sampling, block-average variance estimation, and incremental likelihood updates — are standard tools in Bayesian inference, statistical modeling, and computational simulation, applied here to a real graduate research problem.

---

---

© 2025 Ana Flores — Master's thesis project, Universidad de Guanajuato
