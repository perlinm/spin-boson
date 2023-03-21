#!/usr/bin/env python3
import argparse
import itertools
import multiprocessing
import os
import sys
import time
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


def compute_QFI_vals(
    max_time: float,
    num_spins: int,
    decay_res: float,
    decay_spin: float,
    splitting: float,
    coupling: float,
    initial_state: np.ndarray,
    file_QFI: str,
    status_update: bool = False,
) -> None:
    if status_update:
        print(os.path.relpath(file_QFI))
        sys.stdout.flush()
    times, vals_QFI, vals_QFI_SA = methods.get_QFI_vals(
        max_time,
        num_spins,
        splitting,
        coupling,
        decay_res,
        decay_spin,
        initial_state,
    )
    np.savetxt(file_QFI, np.vstack([times, vals_QFI, vals_QFI_SA]))


def batch_compute_QFI_vals(
    max_time: float,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    splitting: float,
    coupling: float,
    state_keys: Sequence[str],
    data_dir: str,
    num_jobs: int = 1,
    recompute: bool = False,
    status_update: bool = False,
) -> None:
    for state_key in state_keys:
        os.makedirs(get_data_dir(data_dir, state_key), exist_ok=True)

    processes = num_jobs if num_jobs >= 0 else multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        results = []

        for num_spins, decay_res, decay_spin, state_key in itertools.product(
            num_spin_vals, decay_res_vals, decay_spin_vals, state_keys
        ):
            args = (state_key, num_spins, decay_res, decay_spin)
            file_QFI = get_file_path(data_dir, "qfi", *args)

            if not os.path.isfile(file_QFI) or recompute:
                initial_state = get_initial_state[state_key](num_spins)
                job_args = (
                    max_time,
                    num_spins,
                    decay_res,
                    decay_spin,
                    splitting,
                    coupling,
                    initial_state,
                    file_QFI,
                    status_update,
                )
                results.append(pool.apply_async(compute_QFI_vals, args=job_args))

        [result.get() for result in results]


def get_simulation_args(sys_argv: Sequence[str]) -> argparse.Namespace:

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Compute QFI and SA-QFI.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--num_spins", type=int, nargs="+", required=True)
    parser.add_argument("--decay_res", type=float, nargs="+", required=True)
    parser.add_argument("--decay_spin", type=float, nargs="+", required=True)
    parser.add_argument(
        "--state_keys", type=str, nargs="+", choices=get_initial_state.keys(), required=True
    )

    # default physical parameters
    parser.add_argument("--splitting", type=float, default=0)  # eV
    parser.add_argument("--coupling", type=float, default=0.04)  # eV
    parser.add_argument("--max_time", type=float, default=100)  # in femptoseconds

    # default directories
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    default_data_dir = os.path.join(script_dir, "data")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    default_fig_dir = os.path.join(script_dir, "figures")
    parser.add_argument("--fig_dir", type=str, default=default_fig_dir)

    # miscellaneous arguments
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--recompute", action="store_true", default=False)

    args = parser.parse_args(sys_argv[1:])

    # convert time units into hbar/eV
    femptosecond = 1e-15  # seconds
    hbar_over_hartree = 2.41888e-17  # seconds
    hartree_over_eV = 27.2114  # dimensionless
    hbar_over_eV = hbar_over_hartree * hartree_over_eV  # hbar/eV
    args.max_time *= femptosecond / hbar_over_eV

    return args


if __name__ == "__main__":

    start = time.time()

    args = get_simulation_args(sys.argv)
    batch_compute_QFI_vals(
        args.max_time,
        args.num_spins,
        args.decay_res,
        args.decay_spin,
        args.splitting,
        args.coupling,
        args.state_keys,
        args.data_dir,
        args.num_jobs,
        args.recompute,
        status_update=True,
    )

    print("DONE")
    print(time.time() - start)
