"""Experiment D -- real handwritten-digit images.

Twenty transport problems are built from the scikit-learn ``digits`` dataset
(1797 grey-scale 8x8 images): ten pairs of images of the *same* digit label and
ten pairs of *different* labels. The grey levels of each image are normalised
into a probability distribution over the pixel grid, and the cost of moving mass
from one pixel to another is the squared Euclidean distance between pixels.

This is a real-data instance family with a genuinely discrete, sparse support,
which stresses the solvers differently from the dense synthetic instances of
experiment B.

Run with::

    python experiments/run_experiment_d.py
"""

from __future__ import annotations

import numpy as np

from common import BUDGET, RESULTS_DIR, certified_bracket, run_methods, save_json
from otopt import config, data, reference

TOLS = config.TOLERANCES
KEYS = (
    "iterations",
    "time",
    "final_feasibility_error",
    "final_objective_error",
    "final_negative_mass",
    "iter_to_1e-03",
    "time_to_1e-03",
    "iter_to_1e-04",
    "time_to_1e-04",
    "certified_error_bound",
    "certified_error_bound_polished",
    "signed_objective_gap",
    "signed_objective_gap_vs_lower_bound",
    "certificate_is_valid",
    "certified_at_1e-03",
    "certified_at_1e-04",
    "polished_feasibility_error",
)
N_PLANS = 2                     # instances whose plans are stored for plotting


def main():
    instances = data.make_digits_instances(
        n_same=config.DIGITS_PAIRS,
        n_cross=config.DIGITS_CROSS_PAIRS,
        seed=config.DIGITS_SEED,
        eps=config.DIGITS_EPS,
    )
    print(f"[D] {len(instances)} digit pairs")

    meta, runs = [], []
    plans = {}
    for k, inst in enumerate(instances, 1):
        ref = reference.solve_reference(inst.a, inst.b, inst.C)
        bracket = certified_bracket(inst, ref["plan"], ref["objective"])
        f_lower = bracket["lower"]
        meta.append(
            {
                "name": inst.name,
                **inst.meta,
                "m": inst.m,
                "n": inst.n,
                "support_a": int(inst.a.size),
                "support_b": int(inst.b.size),
                "reference_objective": ref["objective"],
                "reference_time": ref["time"],
                "reference_iterations": ref["iterations"],
                "reference_lower_bound": f_lower,
                "reference_upper_bound": bracket["upper"],
                "reference_bracket_width": bracket["width"],
                "reference_polish_feasibility_error": bracket["polish_feasibility_error"],
                "reference_polish_reached_precision": bracket["polish_reached_precision"],
                "reference_bracket_is_valid": bracket["bracket_is_valid"],
                "reference_inside_bracket": bracket["reference_inside"],
                "reference_error_lower_bound": bracket["reference_error_lower_bound"],
                "reference_below_lower": bracket["reference_below_lower"],
                "reference_error_upper_bound": bracket["reference_error_upper_bound"],
            }
        )
        out = run_methods(inst, ref["objective"], TOLS, BUDGET, f_lower=f_lower)
        print(
            f"[D] {k}/{len(instances)} {inst.name}: "
            + ", ".join(f"{m}={o['final_feasibility_error']:.0e}" for m, o in out.items())
        )
        for method, o in out.items():
            runs.append({"instance": inst.name, "method": method, **{key: o[key] for key in KEYS}})

        if k <= N_PLANS:
            plans[f"a_{k}"] = inst.a
            plans[f"b_{k}"] = inst.b
            plans[f"C_{k}"] = inst.C
            plans[f"ref_{k}"] = ref["plan"]
            for method in ("ADMM (fixed rho)", "PDHG"):
                plans[f"{method}_{k}"] = out[method]["plan"]

    save_json(
        {
            "budget": BUDGET,
            "tolerances": list(TOLS),
            "instances": meta,
            "runs": runs,
        },
        RESULTS_DIR / "experiment_d.json",
    )

    # source/target positions are needed for the qualitative figure and are not
    # stored in the JSON (they are positional data), so they go to a small npz.
    pos = {}
    from otopt.data import _image_mass
    from sklearn.datasets import load_digits

    digits = load_digits()
    for k, inst in enumerate(instances[:N_PLANS], 1):
        pa, _ = _image_mass(digits.images[inst.meta["index_a"]], config.DIGITS_EPS)
        pb, _ = _image_mass(digits.images[inst.meta["index_b"]], config.DIGITS_EPS)
        pos[f"pos_a_{k}"] = pa
        pos[f"pos_b_{k}"] = pb
        pos[f"image_a_{k}"] = digits.images[inst.meta["index_a"]]
        pos[f"image_b_{k}"] = digits.images[inst.meta["index_b"]]
    pos.update({key: val for key, val in plans.items() if val is not None})
    np.savez(RESULTS_DIR / "experiment_d_plans.npz", **pos)
    print(f"[saved] {RESULTS_DIR / 'experiment_d_plans.npz'}")


if __name__ == "__main__":
    main()
