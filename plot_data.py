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


def get_fit_exponent(x_vals: Sequence[int], y_vals: Sequence[float]) -> float:
    popt, _ = scipy.optimize.curve_fit(
        poly_func,
        x_vals,
        y_vals,
        bounds=([0, 0, 0], [800000000, 3, 800000000]),
    )
    return popt[1]


def get_all_data(
    state_key: str,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    data_dir: str,
) -> np.ndarray:
    """Retrieve all simulated QFI and SA-QFI data.
    Results are organized into arrays indexed by key = (spin_num, decay_res, decay_spin).
    """
    shape = (len(num_spin_vals), len(decay_res_vals), len(decay_spin_vals))
    vals_QFI = np.zeros(shape, dtype=object)

    for nn, num_spins in enumerate(num_spin_vals):
        for kk, decay_res in enumerate(decay_res_vals):
            for gg, decay_spin in enumerate(decay_spin_vals):
                args = (state_key, num_spins, decay_res, decay_spin)
                file_QFI = get_file_path(data_dir, "qfi", *args)
                vals_QFI[nn, kk, gg] = np.loadtxt(file_QFI)

    return vals_QFI


def extract_maxima(vals_QFI: np.ndarray) -> np.ndarray:
    vals_max = [vals_QFI[idx][1:, :].max(axis=1) for idx in np.ndindex(vals_QFI.shape)]
    return np.array(vals_max).reshape(vals_QFI.shape + (2,))


def get_maxima(
    state_key: str,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    data_dir: str,
) -> np.ndarray:
    """Maximize QFI and SA-QFI data over simulation time.
    Results are organized into arrays indexed by key = (spin_num, decay_res, decay_spin).
    """
    vals_QFI = get_all_data(state_key, num_spin_vals, decay_res_vals, decay_spin_vals, data_dir)
    return extract_maxima(vals_QFI)


def extract_exponents(num_spin_vals: Sequence[int], vals_max: np.ndarray) -> np.ndarray:
    exponent_vals = [
        get_fit_exponent(num_spin_vals, tuple(vals_max[:, kk, gg, qq]))
        for kk in range(vals_max.shape[1])
        for gg in range(vals_max.shape[2])
        for qq in range(vals_max.shape[3])
    ]
    return np.array(exponent_vals).reshape(vals_max.shape[1:])


def get_exponents(
    state_key: str,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    data_dir: str,
) -> np.ndarray:
    """Get the scaling of QFI and SA-QFI with spin number.
    Results are organized into arrays indexed by [decay_res, decay_spin].
    """
    vals_max = get_maxima(state_key, num_spin_vals, decay_res_vals, decay_spin_vals, data_dir)
    return extract_exponents(num_spin_vals, vals_max)


if __name__ == "__main__":
    args = get_simulation_args(sys.argv)
    os.makedirs(args.fig_dir, exist_ok=True)

    for state_key in args.state_keys:
        vals_QFI = get_all_data(
            state_key,
            args.num_spins,
            args.decay_res,
            args.decay_spin,
            args.data_dir,
        )
        vals_max = extract_maxima(vals_QFI)
        vals_exp = extract_exponents(args.num_spins, vals_max)

        print(f"plotting exponents ({state_key})")
        for vals, tag in [(vals_exp[:, :, 0], "qfi"), (vals_exp[:, :, 1], "qfi-SA")]:
            fig, ax = plt.subplots(figsize=(4, 3))
            color_mesh = ax.pcolormesh(args.decay_res, args.decay_spin, vals.T)
            fig.colorbar(color_mesh, label="scaling")
            ax.set_xlabel(r"$\kappa$ [eV]")
            ax.set_ylabel(r"$\gamma$ [eV]")
            plt.tight_layout()

            fig_path = os.path.join(args.fig_dir, f"{tag}_{state_key}.pdf")
            plt.savefig(fig_path)
            plt.close()

        print(f"plotting system size scaling ({state_key})")
        for (kk, kappa), (gg, gamma) in itertools.product(
            enumerate(args.decay_res), enumerate(args.decay_spin)
        ):
            plt.figure(figsize=(4, 3))
            plt.title(rf"$\kappa={kappa}$, $\gamma={gamma}$")
            plt.plot(args.num_spins, vals_max[:, kk, gg, 0], "ko")
            plt.plot(args.num_spins, vals_max[:, kk, gg, 1], "bo")
            plt.xlabel("$N$")
            plt.ylabel("QFI [1/eV$^2$]")
            plt.tight_layout()

            fig_name = f"qfi_{state_key}_k{kappa:.4f}_g{gamma:.4f}.pdf"
            plt.savefig(os.path.join(args.fig_dir, fig_name))
            plt.close()

        print(f"plotting time-series data ({state_key})")
        for (nn, num_spins), (kk, kappa), (gg, gamma) in itertools.product(
            enumerate(args.num_spins), enumerate(args.decay_res), enumerate(args.decay_spin)
        ):
            plt.figure(figsize=(4, 3))
            plt.title(rf"$N={num_spins}$, $\kappa={kappa}$, $\gamma={gamma}$")
            plt.plot(vals_QFI[nn, kk, gg][0], vals_QFI[nn, kk, gg][1], "k-")
            plt.plot(vals_QFI[nn, kk, gg][0], vals_QFI[nn, kk, gg][2], "k--")
            plt.xlabel("time")
            plt.ylabel("QFI [1/eV$^2$]")
            plt.tight_layout()

            fig_name = f"qfi_{state_key}_N{num_spins}_k{kappa:.4f}_g{gamma:.4f}.pdf"
            plt.savefig(os.path.join(args.fig_dir, fig_name))
            plt.close()
