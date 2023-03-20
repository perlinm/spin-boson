#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Sequence

import numpy as np

import methods


def get_data_dir(data_dir: str, state_key: str) -> str:
    return os.path.join(data_dir, state_key)


def get_file_path(
    data_dir: str, prefix: str, state_key: str, num_spins: int, decay_res: float, decay_spin: float
) -> str:
    data_dir = get_data_dir(data_dir, state_key)
    return os.path.join(data_dir, f"{prefix}_N{num_spins}_k{decay_res:.4f}_g{decay_spin:.4f}.txt")


get_initial_state = {
    "ghz": methods.get_ghz_state,
    "dicke-1": lambda num_spins: methods.get_dicke_state(num_spins, 1),
    "dicke-2": lambda num_spins: methods.get_dicke_state(num_spins, 2),
}


def compute_fisher_vals(
    times: np.ndarray,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    splitting: float,
    coupling: float,
    state_key: str,
    data_dir: str,
    status_updates: bool = False,
) -> None:
    os.makedirs(get_data_dir(data_dir, state_key), exist_ok=True)

    for num_spins in num_spin_vals:
        for kk, decay_res in enumerate(decay_res_vals):
            for gg, decay_spin in enumerate(decay_spin_vals):
                if status_updates:
                    print(num_spins, f"{kk}/{len(decay_res_vals)}", f"{gg}/{len(decay_spin_vals)}")

                fisher_vals, scaled_fisher_vals = methods.get_fisher_vals(
                    times,
                    num_spins,
                    splitting,
                    coupling,
                    decay_res,
                    decay_spin,
                    get_initial_state[state_key](num_spins),
                )

                args = (state_key, num_spins, decay_res, decay_spin)
                file_QFI = get_file_path(data_dir, "fisher", *args)
                file_QFI_SA = get_file_path(data_dir, "fisher-SA", *args)
                np.savetxt(file_QFI, fisher_vals)
                np.savetxt(file_QFI_SA, scaled_fisher_vals)


def get_simulation_args(sys_argv: Sequence[str]) -> argparse.Namespace:

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Compute QFI and SA-QFI.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num_spins", type=int, nargs="+", required=True)
    parser.add_argument("--decay_res", type=float, nargs="+", required=True)
    parser.add_argument("--decay_spin", type=float, nargs="+", required=True)
    parser.add_argument("--state_key", type=str, choices=get_initial_state.keys(), required=True)

    # default physical parameters
    parser.add_argument("--splitting", type=float, default=0)  # eV
    parser.add_argument("--coupling", type=float, default=0.04)  # eV
    parser.add_argument("--max_time", type=float, default=100)  # in femptoseconds
    parser.add_argument("--time_points", type=int, default=101)

    # default directories
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    default_data_dir = os.path.join(script_dir, "data")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    default_fig_dir = os.path.join(script_dir, "figures")
    parser.add_argument("--fig_dir", type=str, default=default_fig_dir)

    args = parser.parse_args(sys_argv[1:])
    hartree_per_eV = 27.2114
    autime = 2.41888e-17
    args.max_time *= 1e-15 / (autime * hartree_per_eV)

    return args


if __name__ == "__main__":

    args = get_simulation_args(sys.argv)
    times = np.linspace(0, args.max_time, args.time_points)
    compute_fisher_vals(
        times,
        args.num_spins,
        args.decay_res,
        args.decay_spin,
        args.splitting,
        args.coupling,
        args.state_key,
        args.data_dir,
        status_updates=True,
    )
