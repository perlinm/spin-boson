#!/usr/bin/env python3
"""Script to plot the results of simulations of a spin-boson system."""
import functools
import itertools
import os
import sys
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

import collect_data

DOT_KWARGS = dict(linestyle="-", marker=".")
FIGSIZE = (3.4, 2.4)
SURFACE_FIGSIZE = (2.3, 1.8)

MAX_NUM_SPINS = 20

QFI_OVER_TIME = False
BASE_FIG_DIR = "figures" + ("-qfi-over-time" if QFI_OVER_TIME else "")
DATA_DIR = "data"


if QFI_OVER_TIME:
    QFI_UNITS = " (units of $g^{-1})$"
    QFI_LABEL = r"QFI$/t$" + QFI_UNITS
    MAX_QFI_LABEL = r"$\mathrm{max}_t$ QFI$(t)/t$" + QFI_UNITS
    DOUBLE_MAX_QFI_LABEL = r"$\mathrm{max}_{t,n}$ QFI$(t,n)/t$" + QFI_UNITS

else:
    QFI_UNITS = " (units of $g^{-2}$)"
    QFI_LABEL = "QFI" + QFI_UNITS
    MAX_QFI_LABEL = r"$\mathrm{max}_t$ QFI$(t)$" + QFI_UNITS
    DOUBLE_MAX_QFI_LABEL = r"$\mathrm{max}_{t,n}$ QFI$(t,n)$" + QFI_UNITS


@functools.cache
def get_QFI_data(
    state_key: str,
    decay_res: float,
    decay_spin: float,
    dephasing: bool,
    num_spins: int,
    data_dir: str = DATA_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Get raw simulated QFI data: time and QFI values."""
    args = (state_key, num_spins, decay_res, decay_spin, dephasing)
    file_QFI = collect_data.get_file_path(data_dir, "qfi", *args)
    time, vals = np.loadtxt(file_QFI, unpack=True)
    if QFI_OVER_TIME:
        return time[1:], vals[1:] / time[1:]
    return time, vals


@functools.cache
def get_max_QFI(
    state_key: str,
    decay_res: float,
    decay_spin: float,
    dephasing: bool,
    num_spins: int,
    data_dir: str = DATA_DIR,
) -> float:
    """Get the maximum QFI observed in simulations."""
    if state_key == "dicke-max":
        return max(
            get_max_QFI(f"dicke-{nn}", decay_res, decay_spin, dephasing, num_spins, data_dir)
            for nn in range(num_spins + 1)
        )
    try:
        return get_QFI_data(
            state_key,
            decay_res,
            decay_spin,
            dephasing,
            num_spins,
            data_dir,
        )[1].max()
    except FileNotFoundError:
        raise ValueError("No data available")


@functools.cache
def get_scaling_exponent(
    state_key: str,
    decay_res: float,
    decay_spin: float,
    dephasing: bool,
    num_spin_vals: tuple[int, ...],
    data_dir: str = DATA_DIR,
) -> float:
    """Extract QFI scaling exponents: max_t QFI(t) ~ N^x."""
    min_num_spins = get_min_num_spins(state_key)
    max_QFI = [
        get_max_QFI(state_key, decay_res, decay_spin, dephasing, num_spins)
        for num_spins in num_spin_vals
        if num_spins >= min_num_spins
    ]
    return get_exp_fit_params(num_spin_vals, max_QFI)[0][-1]


def get_min_num_spins(state_key: str) -> int:
    """Get minimum spin number for the given state."""
    if state_key[:6] == "dicke-" and state_key != "dicke-max":
        return int(state_key.strip("dicke-"))
    return 0


def get_exp_fit_params(
    x_vals: Sequence[int], y_vals: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Get the fit parameters (a,b,c) in y ~= a x^c - b."""
    fit_params, fit_cov = scipy.optimize.curve_fit(
        lambda xx, aa, bb, cc: aa * xx**cc - bb,
        x_vals,
        y_vals,
        p0=(y_vals[-1] / x_vals[-1], 0, 1),
        maxfev=10**5,
    )
    return fit_params, fit_cov


def get_fig_dir(subdir: str = "", base_dir: str = BASE_FIG_DIR) -> str:
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
    if state_key == "dicke-max":
        return r"$\max_n$\,D-$n$"
    if "dicke" in state_key:
        return state_key.replace("dicke", "D")
    return state_key


def plot_time_series(decay_vals: Sequence[float], dephasing: bool, silent: bool = False) -> None:
    """
    QFI as a function of time.
    Plots several system sizes.
    Fixes decay constants and initial state.
    """
    fig_dir = get_fig_dir("time_series")
    state_keys = ["ghz", "x-polarized"] + [f"dicke-{nn}" for nn in (1, 2, 5, 10, 15)]
    for state_key, decay_res, decay_spin in itertools.product(state_keys, decay_vals, decay_vals):
        if not silent:
            print(state_key, decay_res, decay_spin)

        plt.figure(figsize=FIGSIZE)
        state_name = r"$|$" + get_state_name(state_key) + r"$\rangle$"
        plt.title(rf"$\kappa/g={decay_res:.1f}$, $\gamma/g={decay_spin:.1f}$, {state_name}")
        for num_spins in [20, 15, 10, 5]:
            if is_invalid(state_key, num_spins):
                continue
            time, vals = get_QFI_data(state_key, decay_res, decay_spin, dephasing, num_spins)
            plt.plot(time, vals, label=rf"$N={num_spins}$")
        plt.xlabel(r"time $\times g$")
        plt.ylabel(QFI_LABEL)
        plt.legend(loc="lower right", framealpha=1)
        plt.tight_layout(pad=0.2)

        fig_name = f"time_{state_key}_k{decay_res:.2f}_g{decay_spin:.2f}"
        if dephasing:
            fig_name += "_z"
        plt.savefig(os.path.join(fig_dir, f"{fig_name}.pdf"))
        plt.close()


def plot_size_scaling(decay_vals: Sequence[float], dephasing: bool, silent: bool = False) -> None:
    """
    max_time QFI(time) as a function of system size.
    Plots several initial states.
    Fixes decay constants.
    """
    fig_dir = get_fig_dir("size_scaling")
    state_keys = ["ghz", "x-polarized"] + [f"dicke-{nn}" for nn in (1, 5, 10)] + ["dicke-max"]
    for decay_res, decay_spin in itertools.product(decay_vals, decay_vals):
        if not silent:
            print(decay_res, decay_spin)

        plt.figure(figsize=FIGSIZE)
        plt.title(rf"$\kappa/g={decay_res:.1f}$, $\gamma/g={decay_spin:.1f}$")
        for state_key in state_keys:
            min_num_spins = get_min_num_spins(state_key)
            num_spin_vals = tuple(range(min_num_spins, MAX_NUM_SPINS + 1))

            max_QFI_vals = [
                get_max_QFI(state_key, decay_res, decay_spin, dephasing, num_spins)
                for num_spins in num_spin_vals
            ]

            label = get_state_name(state_key)
            dot_kwargs = DOT_KWARGS.copy()
            if "max" in state_key:
                dot_kwargs["color"] = "k"
                dot_kwargs["linestyle"] = "--"
            plt.plot(
                num_spin_vals,
                max_QFI_vals,
                label=label,
                **dot_kwargs,  # type:ignore[arg-type]
            )
        plt.xlabel(r"$N$")
        plt.ylabel(MAX_QFI_LABEL)
        plt.legend(loc="best")
        plt.ticklabel_format(scilimits=(-3, 3), useMathText=True)
        plt.tight_layout(pad=0.2)

        fig_name = f"scaling_k{decay_res:.2f}_g{decay_spin:.2f}"
        if dephasing:
            fig_name += "_z"
        plt.savefig(os.path.join(fig_dir, f"{fig_name}.pdf"))
        plt.close()


def plot_dicke_k(decay_vals: Sequence[float], dephasing: bool, silent: bool = False) -> None:
    """
    max_time QFI(time) as a function of Dicke state index.
    Plots several spin decay rates.
    Fixes system size and resonator decay rate.
    """
    fig_dir = get_fig_dir("dicke-k")
    num_spins = MAX_NUM_SPINS
    num_spin_vals = list(range(num_spins + 1))
    for decay_res in decay_vals:
        if not silent:
            print(decay_res)

        plt.figure(figsize=FIGSIZE)
        plt.title(rf"$N={num_spins}$, $\kappa/g={decay_res:.1f}$")
        for decay_spin in decay_vals:
            max_QFI_vals = [
                get_max_QFI(f"dicke-{nn}", decay_res, decay_spin, dephasing, num_spins)
                for nn in num_spin_vals
            ]
            label = rf"$\gamma/g={decay_spin:.1f}$"
            plt.plot(
                num_spin_vals,
                max_QFI_vals,
                label=label,
                **DOT_KWARGS,  # type:ignore[arg-type]
            )
        plt.xlabel(r"D-$n$")
        plt.ylabel(MAX_QFI_LABEL)
        plt.ticklabel_format(scilimits=(-3, 3), useMathText=True)

        # legend
        handles, labels = plt.gca().get_legend_handles_labels()
        leg_title = mpl.lines.Line2D([0], [0], color="w", marker="None", label=r"$\gamma/g$")
        handles = [leg_title] + handles
        plt.legend(handles=handles, loc="upper right", framealpha=1, bbox_to_anchor=(1.2, 1.15))

        plt.tight_layout(pad=0.2)

        fig_name = f"dicke-k_{decay_res:.2f}"
        if dephasing:
            fig_name += "_z"
        plt.savefig(os.path.join(fig_dir, f"{fig_name}.pdf"))
        plt.close()


def plot_dicke_g(decay_vals: Sequence[float], dephasing: bool, silent: bool = False) -> None:
    """
    max_time QFI(time) as a function of Dicke state index.
    Plots several resonator decay rates.
    Fixes system size and spin decay rate.
    """
    fig_dir = get_fig_dir("dicke-g")
    num_spins = MAX_NUM_SPINS
    num_spin_vals = list(range(num_spins + 1))
    for decay_spin in decay_vals:
        if not silent:
            print(decay_spin)

        plt.figure(figsize=FIGSIZE)
        plt.title(rf"$N={num_spins}$, $\gamma/g={decay_spin:.1f}$")
        for decay_res in decay_vals:
            max_QFI_vals = [
                get_max_QFI(f"dicke-{nn}", decay_res, decay_spin, dephasing, num_spins)
                for nn in num_spin_vals
            ]
            label = rf"${decay_res:.1f}$"
            plt.plot(
                num_spin_vals,
                max_QFI_vals,
                label=label,
                **DOT_KWARGS,  # type:ignore[arg-type]
            )
        plt.xlabel(r"D-$n$")
        plt.ylabel(MAX_QFI_LABEL)
        plt.ticklabel_format(scilimits=(-3, 3), useMathText=True)

        # legend
        handles, labels = plt.gca().get_legend_handles_labels()
        leg_title = mpl.lines.Line2D([0], [0], color="w", marker="None", label=r"$\kappa/g$")
        handles = [leg_title] + handles
        plt.legend(handles=handles, loc="upper right", framealpha=1, bbox_to_anchor=(1.2, 1.15))

        plt.tight_layout(pad=0.2)

        fig_name = f"dicke-g_{decay_spin:.2f}"
        if dephasing:
            fig_name += "_z"
        plt.savefig(os.path.join(fig_dir, f"{fig_name}.pdf"))
        plt.close()


def plot_surface_exponents(
    decay_vals: Sequence[float], dephasing: bool, silent: bool = False
) -> None:
    """
    Surface plot of QFI scaling exponent as a function of decay rates.
    Fixes initial state.
    """
    fig_dir = get_fig_dir("surface_exponents")
    state_keys = ["ghz", "x-polarized"] + [f"dicke-{nn}" for nn in range(1, 16)]
    for state_key in state_keys:
        if not silent:
            print(state_key)

        min_num_spins = get_min_num_spins(state_key)
        num_spin_vals = tuple(range(min_num_spins, MAX_NUM_SPINS + 1))

        exponents = [
            [
                get_scaling_exponent(state_key, decay_res, decay_spin, dephasing, num_spin_vals)
                for decay_spin in decay_vals
            ]
            for decay_res in decay_vals
        ]
        if decay_vals[0] == 0:
            exponents[0][0] = np.nan

        fig, ax = plt.subplots(figsize=SURFACE_FIGSIZE)
        plt.title(get_state_name(state_key))
        color_mesh = ax.pcolormesh(decay_vals, decay_vals, np.array(exponents).T)
        fig.colorbar(color_mesh, label="scaling exponent")
        ax.set_xlabel(r"$\kappa/g$")
        ax.set_ylabel(r"$\gamma/g$")
        plt.xticks(decay_vals)
        plt.tight_layout(pad=0.2)

        fig_name = f"exponents_{state_key}"
        if dephasing:
            fig_name += "_z"
        plt.savefig(os.path.join(fig_dir, f"{fig_name}.pdf"))
        plt.close()


def plot_surface_maxima(decay_vals: Sequence[float], dephasing: bool) -> None:
    """
    Surface plot of QFI maximum as a function of decay rates.
    Maximizes over intial Dicke state index.
    Fixes spin number.
    """
    fig_dir = get_fig_dir()
    num_spins = MAX_NUM_SPINS
    state_keys = [f"dicke-{nn}" for nn in range(num_spins + 1)]
    maxima = [
        [
            max(
                [
                    get_max_QFI(state_key, decay_res, decay_spin, dephasing, num_spins)
                    for state_key in state_keys
                ]
            )
            for decay_spin in decay_vals
        ]
        for decay_res in decay_vals
    ]
    if decay_vals[0] == 0:
        maxima[0][0] = np.nan

    fig, ax = plt.subplots(figsize=SURFACE_FIGSIZE)
    plt.title(rf"$N={num_spins}$")
    color_mesh = ax.pcolormesh(
        decay_vals, decay_vals, np.array(maxima).T, norm=mpl.colors.LogNorm()
    )
    fig.colorbar(color_mesh, label=DOUBLE_MAX_QFI_LABEL)
    ax.set_xlabel(r"$\kappa/g$")
    ax.set_ylabel(r"$\gamma/g$")
    plt.tight_layout(pad=0.2)

    fig_name = f"{plot}_N{num_spins}"
    if dephasing:
        fig_name += "_z"
    plt.savefig(os.path.join(fig_dir, f"{fig_name}.pdf"))
    plt.close()


def plot_surface_dicke(decay_vals: Sequence[float], dephasing: bool) -> None:
    """
    Surface plot of optimal Dicke state index as a function of decay rates.
    Fixes spin number.
    """
    fig_dir = get_fig_dir()
    num_spins = MAX_NUM_SPINS
    state_keys = [f"dicke-{nn}" for nn in range(num_spins + 1)]
    dicke_index = [
        [
            np.argmax(
                [
                    get_max_QFI(state_key, decay_res, decay_spin, dephasing, num_spins)
                    for state_key in state_keys
                ]
            )
            for decay_spin in decay_vals
        ]
        for decay_res in decay_vals
    ]
    if decay_vals[0] == 0:
        dicke_index[0][0] = np.nan

    fig, ax = plt.subplots(figsize=SURFACE_FIGSIZE)
    plt.title(rf"$N={num_spins}$")
    color_mesh = ax.pcolormesh(decay_vals, decay_vals, np.array(dicke_index).T)
    fig.colorbar(color_mesh, label=r"D-$n$")
    ax.set_xlabel(r"$\kappa/g$")
    ax.set_ylabel(r"$\gamma/g$")
    plt.tight_layout(pad=0.2)

    fig_name = f"{plot}_N{num_spins}"
    if dephasing:
        fig_name += "_z"
    plt.savefig(os.path.join(fig_dir, f"{fig_name}.pdf"))
    plt.close()


if __name__ == "__main__":
    plot = sys.argv[1] if len(sys.argv) > 1 else ""
    silent = False

    dephasing = False
    decay_vals = list(np.arange(0.2, 1.01, 0.2))

    font_size = 10
    params = {
        "font.family": "serif",
        "font.serif": "Computer Modern",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
    }
    plt.rcParams.update(params)

    if plot == "time_series":
        plot_time_series(decay_vals, dephasing, silent)
    if plot == "size_scaling":
        plot_size_scaling(decay_vals, dephasing, silent)
    if plot == "dicke-k":
        plot_dicke_k(decay_vals, dephasing, silent)
    if plot == "dicke-g":
        plot_dicke_g(decay_vals, dephasing, silent)

    font_size = 8
    params = {
        "font.family": "serif",
        "font.serif": "Computer Modern",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
    }
    plt.rcParams.update(params)

    if plot == "surface_exponents":
        plot_surface_exponents(decay_vals, dephasing, silent)
    if plot == "surface_maxima":
        plot_surface_maxima(decay_vals, dephasing)
    if plot == "surface_dicke":
        plot_surface_dicke(decay_vals, dephasing)
