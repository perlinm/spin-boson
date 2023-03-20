#!/usr/bin/env python3
import os
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy

from collect_data import get_file_path, get_simulation_args


def poly_func(x: float, a: float, b: float, c: float) -> float:
    return a * x**b + c


def get_fit_exponent(x_vals: Sequence[float | int], y_vals: Sequence[float | int]) -> float:
    popt, _ = scipy.optimize.curve_fit(
        poly_func,
        x_vals,
        y_vals,
        bounds=([0, 0, 0], [800000000, 3, 800000000]),
    )
    return popt[1]


def first_local_maximum(vals: np.ndarray) -> float:
    for val_a, val_b in zip(vals[:-1], vals[1:]):
        if val_a > val_b:
            return val_a
    return vals[-1]


def compute_exponents(
    state_key: str,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    grid_shape = (len(decay_res_vals), len(decay_spin_vals))
    max_fisher_exps = np.zeros(grid_shape)
    scaled_max_fisher_exps = np.zeros(grid_shape)

    for kk, decay_res in enumerate(decay_res_vals):
        for gg, decay_spin in enumerate(decay_spin_vals):
            max_QFI = []
            max_QFI_SA = []

            for num_spins in num_spin_vals:
                args = (state_key, num_spins, decay_res, decay_spin)
                file_QFI = get_file_path(data_dir, "fisher", *args)
                file_QFI_SA = get_file_path(data_dir, "fisher-SA", *args)
                fisher_vals = np.loadtxt(file_QFI)
                scaled_fisher_vals = np.loadtxt(file_QFI_SA)

                max_QFI.append(first_local_maximum(fisher_vals))
                max_QFI_SA.append(first_local_maximum(scaled_fisher_vals))

            max_fisher_exps[kk, gg] = get_fit_exponent(num_spin_vals, max_QFI)
            scaled_max_fisher_exps[kk, gg] = get_fit_exponent(num_spin_vals, max_QFI_SA)

    return max_fisher_exps, scaled_max_fisher_exps


if __name__ == "__main__":
    args = get_simulation_args(sys.argv)
    os.makedirs(args.fig_dir, exist_ok=True)

    for state_key in args.state_keys:
        max_fisher_exps, scaled_max_fisher_exps = compute_exponents(
            state_key,
            args.num_spins,
            args.decay_res,
            args.decay_spin,
            args.data_dir,
        )

        fig, ax = plt.subplots(figsize=(4, 3))
        color_mesh = ax.pcolormesh(
            args.decay_res,
            args.decay_spin,
            scaled_max_fisher_exps.T,
        )
        fig.colorbar(color_mesh, label="scaling")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\gamma$")
        plt.tight_layout()

        fig_path = os.path.join(args.fig_dir, f"{state_key}.pdf")
        plt.savefig(fig_path)
