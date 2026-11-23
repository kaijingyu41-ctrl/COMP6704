"""Central configuration: paths, seeds and hyper-parameters.

Every number that can influence a reported result lives here, so that the whole
study can be reproduced from a single place.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# --------------------------------------------------------------------- seeds
SEEDS = (0, 1, 2, 3, 4)

# ---------------------------------------------------------------- evaluation
# Two accuracy levels at which "time to solution" is reported.
TOLERANCES = (1e-3, 1e-4)

# ------------------------------------------------------------ experiment B
SIZES = (20, 50, 100, 200)
MASS_KINDS = ("uniform", "dirichlet0.2")
DIRICHLET_ALPHA = 0.2

# ---------------------------------------------------------- solver budgets
# A run stops as soon as either budget is exhausted. The budget is identical for
# every method so that the comparison is fair.
MAX_ITER = 50000
TIME_LIMIT = 60.0
LOG_EVERY = 25          # record a diagnostic point every this many iterations
HISTORY_POINTS = 150    # raw traces are down-sampled to this many points on disk

# --------------------------------------------------------------------- ADMM
# Default penalty, frozen from the six-instance pilot study of experiment C
# (seeds 10-15, disjoint from the evaluation seeds). On that pilot rho=100
# solves all six instances at 1e-4 with the best worst-case accuracy, while
# rho=10 fails one of them; see Section 5.5 of the report.
ADMM_RHO = 100.0
ADMM_RHO_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
ADMM_RESBAL_MU = 10.0               # residual-balancing trigger factor
ADMM_RESBAL_TAU = 2.0               # rho scaling factor
ADMM_RESBAL_EVERY = 25              # check the residuals every this many iterations
ADMM_RESBAL_MAX_ADAPT = 1000        # stop adapting rho after this iteration

# --------------------------------------------------------------------- PDHG
PDHG_ETA = 0.99                     # tau = sigma = eta / sqrt(m + n)
PDHG_ETA_GRID = (0.3, 0.6, 0.99)
PDHG_RESTART_EVERY = 200            # restart check interval
PDHG_RESTART_BETA = 0.10            # required relative decrease of the residual

# ------------------------------------------------------------ experiment D
DIGITS_PAIRS = 10                   # same-class pairs
DIGITS_CROSS_PAIRS = 10             # different-class pairs
DIGITS_SEED = 0
DIGITS_EPS = 1e-6                   # pixels with less mass than this are dropped

# ------------------------------------------------------------ plotting style
FIGURE_DPI = 200
