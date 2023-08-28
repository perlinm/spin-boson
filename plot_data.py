#!/usr/bin/env python3
"""
Contents: Script to plot the results of simulations of a spin-boson system.
Author: Michael A. Perlin (2023)
"""
import functools
import itertools
import os
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy

import collect_data


DEFAULT_DATA_DIR = "data"


@functools.cache
def get_QFI_data(
    state_key: str,
    decay_res: float,
    decay_spin: float,
    num_spins: int,
    data_dir: str = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Get raw simulated QFI data.

    Simulation data is organized into a 2-D array `data` of shape (2, num_time_points),
    where `data[:, 0]` are simulation times, and `data[:, 1]` are values of QFI.
    """
    args = (state_key, num_spins, decay_res, decay_spin)
    file_QFI = collect_data.get_file_path(data_dir, "qfi", *args)
    return np.loadtxt(file_QFI, unpack=True)


@functools.cache
def get_max_QFI(
    state_key: str,
    decay_res: float,
    decay_spin: float,
    num_spins: int,
    data_dir: str = DEFAULT_DATA_DIR,
) -> float:
    """Get the maximum QFI observed in simulations."""
    return get_QFI_data(state_key, decay_res, decay_spin, num_spins, data_dir)[1].max()


@functools.cache
def get_scaling_exponent(
    state_key: str,
    decay_res: float,
    decay_spin: float,
    num_spin_vals: tuple[int, ...],
    data_dir: str = DEFAULT_DATA_DIR,
) -> float:
    """Extract QFI scaling exponents: max_t QFI(t) ~ N^x."""
    max_QFI = [
        get_max_QFI(state_key, decay_res, decay_spin, num_spins) for num_spins in num_spin_vals
    ]
    return get_exp_fit_params(num_spin_vals, max_QFI)[-1]


def get_exp_fit_params(x_vals: Sequence[int], y_vals: Sequence[float]) -> float:
    """Get the fit parameters (a, b) in y ~= a (x-b)^c."""
    fit_params, _ = scipy.optimize.curve_fit(
        lambda xx, aa, bb: aa * xx**bb,
        x_vals,
        y_vals,
    )
    return fit_params


def get_fig_dir(subdir: str, base_dir: str = "figues") -> str:
    """Make and return a figure directory."""
    fig_dir = os.path.join(base_dir, subdir)
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def is_invalid(state_key: str, num_spins: int) -> bool:
    """Is the given combination of initial state and number of spins invalid?"""
    return "dicke" in state_key and int(state_key.split("-")[1]) > num_spins


def get_state_name(state_key: str) -> str:
    """Get the name of a given initial state."""
    if state_key == "ghz":
        return "GHZ"
    if state_key == "x-polarized":
        return "X"
    if "dicke" in state_key:
        return state_key.replace("dicke", "D")
    return state_key


if __name__ == "__main__":
    plot = sys.argv[1] if len(sys.argv) > 1 else None
    show_progress = True

    fig_dir = "figures"
    figsize = (4, 3)
    decay_vals = np.arange(0.2, 3.01, 0.2)

    """
    QFI as a function of time.
    Plots several system sizes.
    Fixes decay constants and initial state.
    """
    if plot == "time_series":
        fig_dir = get_fig_dir(plot)
        state_keys = ["ghz", "x-polarized"] + [f"dicke-{nn}" for nn in (1, 2, 5, 10, 15)]
        for state_key, decay_res, decay_spin in itertools.product(
            state_keys, decay_vals, decay_vals
        ):
            if show_progress:
                print(state_key, decay_res, decay_spin)

            plt.figure(figsize=figsize)
            state_name = r"$|$" + get_state_name(state_key) + r"$\rangle$"
            plt.title(rf"$\kappa/g={decay_res:.2f}$, $\gamma/g={decay_spin:.2f}$, {state_name}")
            for num_spins in [20, 15, 10, 5]:
                if is_invalid(state_key, num_spins):
                    continue
                time, vals = get_QFI_data(state_key, decay_res, decay_spin, num_spins)
                plt.plot(time, vals, label=rf"N={num_spins}")
            plt.xlabel(r"time $\times g$")
            plt.ylabel(r"QFI $\times g^2$")
            plt.legend(loc="best")
            plt.tight_layout(pad=0.1)

            fig_name = f"time_{state_key}_k{decay_res:.2f}_g{decay_spin:.2f}.pdf"
            plt.savefig(os.path.join(fig_dir, fig_name))
            plt.close()

    """
    max_time QFI(time) as a function of system size.
    Plots several initial states.
    Fixes decay constants.
    """
    if plot == "size_scaling":
        fig_dir = get_fig_dir(plot)
        state_keys = ["ghz", "x-polarized"] + [f"dicke-{nn}" for nn in (1, 2, 5, 10)]
        for decay_res, decay_spin in itertools.product(decay_vals, decay_vals):
            if show_progress:
                print(decay_res, decay_spin)

            plt.figure(figsize=figsize)
            plt.title(rf"$\kappa/g={decay_res:.2f}$, $\gamma/g={decay_spin:.2f}$")
            for state_key in state_keys:
                num_spin_vals = [
                    num_spins for num_spins in range(21) if not is_invalid(state_key, num_spins)
                ]
                max_QFI_vals = [
                    get_max_QFI(state_key, decay_res, decay_spin, num_spins)
                    for num_spins in num_spin_vals
                ]
                plt.plot(num_spin_vals, max_QFI_vals, "o", label=get_state_name(state_key))
            plt.xlabel(r"$N$")
            plt.ylabel(r"$\mathrm{max}_t$ QFI$(t)$ $\times g^2$")
            plt.legend(loc="best")
            plt.tight_layout(pad=0.1)

            fig_name = f"scaling_k{decay_res:.2f}_g{decay_spin:.2f}.pdf"
            plt.savefig(os.path.join(fig_dir, fig_name))
            plt.close()

    """
    max_time QFI(time) as a function of Dicke state index.
    Plots several spin decay rates.
    Fixes system size and resonator decay rate.
    """
    if plot == "dicke-k":
        fig_dir = get_fig_dir(plot)
        num_spins = 20
        num_spin_vals = list(range(num_spins + 1))
        decay_vals = np.arange(0.4, 2.01, 0.4)
        for decay_res in decay_vals:
            if show_progress:
                print(decay_res)

            plt.figure(figsize=figsize)
            plt.title(rf"$N={num_spins}$, $\kappa/g={decay_res:.2f}$")
            for decay_spin in decay_vals:
                max_QFI_vals = [
                    get_max_QFI(f"dicke-{nn}", decay_res, decay_spin, num_spins)
                    for nn in num_spin_vals
                ]
                plt.plot(num_spin_vals, max_QFI_vals, "o", label=rf"$\gamma/g={decay_spin:.2f}$")
            plt.xlabel(r"D-$n$")
            plt.ylabel(r"$\mathrm{max}_t$ QFI$(t)$ $\times g^2$")
            plt.legend(loc="best", framealpha=1)
            plt.tight_layout(pad=0.1)

            fig_name = f"dicke-k_{decay_res:.2f}.pdf"
            plt.savefig(os.path.join(fig_dir, fig_name))
            plt.close()

    """
    max_time QFI(time) as a function of Dicke state index.
    Plots several resonator decay rates.
    Fixes system size and spin decay rate.
    """
    if plot == "dicke-g":
        fig_dir = get_fig_dir(plot)
        num_spins = 20
        num_spin_vals = list(range(num_spins + 1))
        decay_vals = np.arange(0.4, 2.01, 0.4)
        for decay_spin in decay_vals:
            if show_progress:
                print(decay_spin)

            plt.figure(figsize=figsize)
            plt.title(rf"$N={num_spins}$, $\gamma/g={decay_spin:.2f}$")
            for decay_res in decay_vals:
                max_QFI_vals = [
                    get_max_QFI(f"dicke-{nn}", decay_res, decay_spin, num_spins)
                    for nn in num_spin_vals
                ]
                plt.plot(num_spin_vals, max_QFI_vals, "o", label=rf"$\kappa/g={decay_res:.2f}$")
            plt.xlabel(r"D-$n$")
            plt.ylabel(r"$\mathrm{max}_t$ QFI$(t)$ $\times g^2$")
            plt.legend(loc="best", framealpha=1)
            plt.tight_layout(pad=0.1)

            fig_name = f"dicke-g_{decay_spin:.2f}.pdf"
            plt.savefig(os.path.join(fig_dir, fig_name))
            plt.close()

    """
    QFI scaling exponent as a function of decay rates.
    Fixes initial state.
    """
    if plot == "exponents":
        fig_dir = get_fig_dir(plot)
        state_keys = ["ghz", "x-polarized"] + [f"dicke-{nn}" for nn in range(1, 16)]
        for state_key in state_keys:
            if show_progress:
                print(state_key)

            fig, ax = plt.subplots(figsize=figsize)
            plt.title(get_state_name(state_key))

            min_num_spins, max_num_spins = 10, 20
            if "dicke" in state_key:
                min_num_spins = max(min_num_spins, int(state_key.split("-")[1]))
            num_spin_vals = tuple(range(min_num_spins, max_num_spins + 1))
            exponents = [
                [
                    get_scaling_exponent(state_key, decay_res, decay_spin, num_spin_vals)
                    for decay_spin in decay_vals
                ]
                for decay_res in decay_vals
            ]

            color_mesh = ax.pcolormesh(decay_vals, decay_vals, np.array(exponents).T)
            fig.colorbar(color_mesh, label="scaling exponent")
            ax.set_xlabel(r"$\kappa/g$")
            ax.set_ylabel(r"$\gamma/g$")
            plt.tight_layout()

            fig_name = f"exponents_{state_key}.pdf"
            plt.savefig(os.path.join(fig_dir, fig_name))
            plt.close()
