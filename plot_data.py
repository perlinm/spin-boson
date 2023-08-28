#!/usr/bin/env python3
"""
Contents: Script to plot the results of simulations of a spin-boson system.
Author: Michael A. Perlin (2023)
"""
import itertools
import os
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy

from collect_data import get_file_path, get_simulation_args


def get_all_data(
    state_key_vals: Sequence[str],
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    data_dir: str,
) -> np.ndarray:
    """Retrieve all simulated QFI data.

    Results are organized into arrays indexed by
    [state_key, decay_res, decay_spin, spin_num].

    Each choice of keys yields a 2-D array `data` of shape (2, num_time_points),
    where `data[:, 0]` are simulation times, and `data[:, 1]` are values of QFI.
    """
    shape = (len(state_key_vals), len(decay_res_vals), len(decay_spin_vals), len(num_spin_vals))
    vals_QFI = np.zeros(shape, dtype=object)

    for ss, state_key in enumerate(state_key_vals):
        for kk, decay_res in enumerate(decay_res_vals):
            for gg, decay_spin in enumerate(decay_spin_vals):
                for nn, num_spins in enumerate(num_spin_vals):
                    args = (state_key, num_spins, decay_res, decay_spin)
                    file_QFI = get_file_path(data_dir, "qfi", *args)
                    vals_QFI[ss, kk, gg, nn] = np.loadtxt(file_QFI)

    return vals_QFI


def extract_maxima(vals_QFI: np.ndarray) -> np.ndarray:
    """Maximize QFI data over simulation time.

    Results are organized into arrays indexed by
    [state_key, decay_res, decay_spin, spin_num].
    """
    vals_max = [vals_QFI[idx][:, 1].max() for idx in np.ndindex(vals_QFI.shape)]
    return np.array(vals_max).reshape(vals_QFI.shape)


def extract_exponents(num_spin_vals: Sequence[int], vals_max: np.ndarray) -> np.ndarray:
    """Extract QFI scaling exponents.

    Specifically, compute the exponent `b` in `max_time QFI(time) ~= a num_spins^b`.

    Results are organized into arrays indexed by
    [state_key, decay_res, decay_spin].
    """
    shape = vals_max.shape[:-1]
    exponent_vals = [
        get_exp_fit_params(num_spin_vals, vals_max[ss_kk_gg])[1] for ss_kk_gg in np.ndindex(shape)
    ]
    return np.array(exponent_vals).reshape(shape)


def get_exp_fit_params(x_vals: Sequence[int], y_vals: Sequence[float]) -> float:
    """Get the fit parameters (a, b) in y ~= a x^b."""
    fit_params, _ = scipy.optimize.curve_fit(
        lambda xx, aa, bb: aa * xx**bb,
        x_vals,
        y_vals,
    )
    return fit_params


if __name__ == "__main__":
    # parse arguments
    args = get_simulation_args(sys.argv)

    # make figure directories
    for subdir in ["scaling", "surface_max", "surface_exp", "time_series"]:
        os.makedirs(os.path.join(args.fig_dir, subdir), exist_ok=True)

    figsize = (4, 3)

    # read in all data
    vals_QFI = get_all_data(
        args.state_keys,
        args.num_spins,
        args.decay_res,
        args.decay_spin,
        args.data_dir,
    )
    vals_max = extract_maxima(vals_QFI)
    vals_exp = extract_exponents(args.num_spins, vals_max)

    for ss, state_key in enumerate(args.state_keys):
        print(f"making scaling exponent surface plots ({state_key})")
        fig, ax = plt.subplots(figsize=figsize)
        color_mesh = ax.pcolormesh(args.decay_res, args.decay_spin, vals_exp[ss].T)
        fig.colorbar(color_mesh, label="scaling exponent")
        ax.set_xlabel(r"$\kappa/g$")
        ax.set_ylabel(r"$\gamma/g$")
        plt.tight_layout()

        fig_name = f"qfi_{state_key}.pdf"
        plt.savefig(os.path.join(args.fig_dir, "surface_exp", fig_name))
        plt.close()

    for ss, state_key in enumerate(args.state_keys):
        print(f"making max QFI surface plots ({state_key})")
        for nn, num_spins in enumerate(args.num_spins):
            fig, ax = plt.subplots(figsize=figsize)
            color_mesh = ax.pcolormesh(args.decay_res, args.decay_spin, vals_max[ss, :, :, nn].T)
            fig.colorbar(color_mesh, label=r"QFI $\times g^2$")
            ax.set_xlabel(r"$\kappa/g$")
            ax.set_ylabel(r"$\gamma/g$")
            plt.tight_layout()

            fig_name = f"qfi_{state_key}_N{num_spins}.pdf"
            plt.savefig(os.path.join(args.fig_dir, "surface_max", fig_name))
            plt.close()

    print("making plots of max QFI vs. system size")
    for (kk, kappa), (gg, gamma) in itertools.product(
        enumerate(args.decay_res), enumerate(args.decay_spin)
    ):
        plt.figure(figsize=figsize)
        plt.title(rf"$\kappa/g={kappa}$, $\gamma/g={gamma}$")
        for ss, state_key in enumerate(args.state_keys):
            plt.plot(args.num_spins, vals_max[ss, kk, gg, :], "ko", label=state_key)
        plt.xlabel("$N$")
        plt.ylabel(r"QFI $\times g^2$")
        plt.legend(loc="best")
        plt.tight_layout()

        fig_name = f"qfi_k{kappa:.2f}_g{gamma:.2f}.pdf"
        plt.savefig(os.path.join(args.fig_dir, "scaling", fig_name))
        plt.close()

    if args.plot_time_series:
        for ss, state_key in enumerate(args.state_keys):
            print(f"plotting QFI vs. time ({state_key})")
            for (nn, num_spins), (kk, kappa), (gg, gamma) in itertools.product(
                enumerate(args.num_spins), enumerate(args.decay_res), enumerate(args.decay_spin)
            ):
                times, vals = vals_QFI[ss, kk, gg, nn].T
                plt.figure(figsize=figsize)
                plt.title(rf"$N={num_spins}$, $\kappa/g={kappa}$, $\gamma/g={gamma}$")
                plt.plot(times, vals, "k-")
                plt.xlabel(r"time $\times g$")
                plt.ylabel(r"QFI $\times g^2$")
                plt.tight_layout()

                fig_name = f"qfi_{state_key}_N{num_spins}_k{kappa:.2f}_g{gamma:.2f}.pdf"
                plt.savefig(os.path.join(args.fig_dir, "time_series", fig_name))
                plt.close()
