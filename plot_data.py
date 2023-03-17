#!/usr/bin/env python3
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy

from collect_data import gamma_vals, get_file_path, get_initial_state, kappa_vals, spin_num_vals


fig_dir = "./figures"


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
    spin_num_vals: Sequence[int],
    num_kappa_vals: int,
    num_gamma_vals: int,
    state_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    grid_shape = (num_kappa_vals, num_gamma_vals)
    max_fisher_exps = np.zeros(grid_shape)
    scaled_max_fisher_exps = np.zeros(grid_shape)

    for kk in range(num_kappa_vals):
        for gg in range(num_gamma_vals):
            max_QFI = []
            max_QFI_SA = []

            for num_spins in spin_num_vals:
                vals_file = get_file_path("fisher", state_key, num_spins, kk, gg)
                scaled_vals_file = get_file_path("scaled_fisher", state_key, num_spins, kk, gg)
                fisher_vals = np.loadtxt(vals_file)
                scaled_fisher_vals = np.loadtxt(scaled_vals_file)

                max_QFI.append(first_local_maximum(fisher_vals))
                max_QFI_SA.append(first_local_maximum(scaled_fisher_vals))

            max_fisher_exps[kk, gg] = get_fit_exponent(spin_num_vals, max_QFI)
            scaled_max_fisher_exps[kk, gg] = get_fit_exponent(spin_num_vals, max_QFI_SA)

    return max_fisher_exps, scaled_max_fisher_exps


if __name__ == "__main__":
    os.makedirs(fig_dir, exist_ok=True)

    for state_key in get_initial_state.keys():
        max_fisher_exps, scaled_max_fisher_exps = compute_exponents(
            spin_num_vals, len(kappa_vals), len(gamma_vals), state_key
        )

        fig, ax = plt.subplots(figsize=(4, 3))
        color_mesh = ax.pcolormesh(kappa_vals, gamma_vals, scaled_max_fisher_exps.T)
        fig.colorbar(color_mesh, label="scaling")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\gamma$")
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, f"{state_key}.pdf")
        plt.savefig(fig_path)
