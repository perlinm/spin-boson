#!/usr/bin/env python3
import itertools
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


def get_all_data(
    state_key: str,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve all simulated QFI and SA-QFI data.
    Results are organized into arrays indexed by [spin_num, decay_res, decay_spin, time].
    """
    shape = (len(num_spin_vals), len(decay_res_vals), len(decay_spin_vals))
    vals_QFI = np.zeros(shape, dtype=object)
    vals_QFI_SA = np.zeros(shape, dtype=object)

    for nn, num_spins in enumerate(num_spin_vals):
        for kk, decay_res in enumerate(decay_res_vals):
            for gg, decay_spin in enumerate(decay_spin_vals):
                args = (state_key, num_spins, decay_res, decay_spin)
                file_QFI = get_file_path(data_dir, "fisher", *args)
                file_QFI_SA = get_file_path(data_dir, "fisher-SA", *args)
                vals_QFI[nn, kk, gg] = np.loadtxt(file_QFI)
                vals_QFI_SA[nn, kk, gg] = np.loadtxt(file_QFI_SA)

    vals_QFI = np.concatenate(vals_QFI.flatten()).reshape(shape + (-1,))
    vals_QFI_SA = np.concatenate(vals_QFI_SA.flatten()).reshape(shape + (-1,))
    return vals_QFI, vals_QFI_SA


def get_first_maxima(
    state_key: str,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the first maxima from all QFI and SA-QFI data.
    Results are organized into arrays indexed by [spin_num, decay_res, decay_spin].
    """
    vals_QFI, vals_QFI_SA = get_all_data(
        state_key, num_spin_vals, decay_res_vals, decay_spin_vals, data_dir
    )

    def extract_first_maximum(time_series_vals: np.ndarray) -> np.ndarray:
        maxima_vals = [
            first_local_maximum(time_series_vals[nn, kk, gg, :])
            for nn in range(len(num_spin_vals))
            for kk in range(len(decay_res_vals))
            for gg in range(len(decay_spin_vals))
        ]
        shape = (len(num_spin_vals), len(decay_res_vals), len(decay_spin_vals))
        return np.array(maxima_vals).reshape(shape)

    return extract_first_maximum(vals_QFI), extract_first_maximum(vals_QFI_SA)


def get_exponents(
    state_key: str,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the scaling of QFI and SA-QFI with spin number.
    Results are organized into arrays indexed by [decay_res, decay_spin].
    """
    vals_QFI, vals_QFI_SA = get_first_maxima(
        state_key, num_spin_vals, decay_res_vals, decay_spin_vals, data_dir
    )

    def extract_exponents(maxima_vals: np.ndarray) -> np.ndarray:
        exponent_vals = [
            get_fit_exponent(num_spin_vals, tuple(maxima_vals[:, kk, gg]))
            for kk in range(len(decay_res_vals))
            for gg in range(len(decay_spin_vals))
        ]
        shape = (len(decay_res_vals), len(decay_spin_vals))
        return np.array(exponent_vals).reshape(shape)

    return extract_exponents(vals_QFI), extract_exponents(vals_QFI_SA)


if __name__ == "__main__":
    args = get_simulation_args(sys.argv)
    os.makedirs(args.fig_dir, exist_ok=True)

    for state_key in args.state_keys:

        # plot time-series data
        vals_QFI, vals_QFI_SA = get_all_data(
            state_key,
            args.num_spins,
            args.decay_res,
            args.decay_spin,
            args.data_dir,
        )
        times = np.linspace(0, args.max_time, args.time_points)
        for vals, tag in [(vals_QFI, "qfi"), (vals_QFI_SA, "sa-qfi")]:
            for (nn, num_spins), (kk, kappa), (gg, gamma) in itertools.product(
                enumerate(args.num_spins), enumerate(args.decay_res), enumerate(args.decay_spin)
            ):
                plt.figure()
                plt.title(rf"$N={num_spins}$, $\kappa={kappa}$, $\gamma={gamma}$")
                plt.plot(times, vals[nn, kk, gg, :])
                plt.xlabel("time")
                plt.xlabel(tag.upper())
                plt.tight_layout()

                fig_name = f"{tag}_{state_key}_N{num_spins}_k{kappa:.4f}_g{gamma:.4f}.pdf"
                plt.savefig(os.path.join(args.fig_dir, fig_name))
                plt.close()

        # plot exponents
        vals_QFI, vals_QFI_SA = get_exponents(
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
            vals_QFI_SA.T,
        )
        fig.colorbar(color_mesh, label="scaling")
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$\gamma$")
        plt.tight_layout()

        fig_path = os.path.join(args.fig_dir, f"{state_key}.pdf")
        plt.savefig(fig_path)
        plt.close()
