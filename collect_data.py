#!/usr/bin/env python3
import argparse
import itertools
import multiprocessing
import os
import sys
import time
from typing import Callable, Dict, Sequence

import numpy as np

import methods_PS as methods


def get_data_dir(data_dir: str, state_key: str) -> str:
    return os.path.join(data_dir, state_key)


def get_file_path(
    data_dir: str, prefix: str, state_key: str, num_spins: int, decay_res: float, decay_spin: float
) -> str:
    data_dir = get_data_dir(data_dir, state_key)
    return os.path.join(data_dir, f"{prefix}_N{num_spins}_k{decay_res:.2f}_g{decay_spin:.2f}.txt")


get_initial_state: Dict[str, Callable[[int], np.ndarray]] = {
    "ghz": methods.get_ghz_state,
    "dicke-1": lambda num_spins: methods.get_dicke_state(num_spins, 1, boson_dim=2),
    "dicke-2": lambda num_spins: methods.get_dicke_state(num_spins, 2, boson_dim=3),
}


def compute_QFI_vals(
    times: Sequence[float],
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
    vals_QFI, vals_QFI_SA = methods.get_QFI_vals(
        times,
        num_spins,
        splitting,
        coupling,
        decay_res,
        decay_spin,
        initial_state,
    )
    np.savetxt(file_QFI, np.vstack([times, vals_QFI, vals_QFI_SA]))


def batch_compute_QFI_vals(
    times: Sequence[float],
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

    if num_jobs > 1:
        processes = num_jobs if num_jobs >= 0 else multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=processes)
        results = []

    for num_spins, decay_res, decay_spin, state_key in itertools.product(
        num_spin_vals, decay_res_vals, decay_spin_vals, state_keys
    ):
        args = (state_key, num_spins, decay_res, decay_spin)
        file_QFI = get_file_path(data_dir, "qfi", *args)

        if not os.path.isfile(file_QFI) or recompute:
            initial_state = get_initial_state[state_key](num_spins)
            job_args = (
                times,
                num_spins,
                decay_res,
                decay_spin,
                splitting,
                coupling,
                initial_state,
                file_QFI,
                status_update,
            )
            if num_jobs > 1:
                results.append(pool.apply_async(compute_QFI_vals, args=job_args))
            else:
                compute_QFI_vals(*job_args)

    if num_jobs > 1:
        pool.close()
        pool.join()


def get_simulation_args(sys_argv: Sequence[str]) -> argparse.Namespace:

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Compute QFI and SA-QFI.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "--state_keys", type=str, nargs="+", choices=get_initial_state.keys(), required=True
    )
    parser.add_argument("--num_spins", type=int, nargs="+", required=True)

    parser.add_argument("--decay", type=float, nargs="+")
    parser.add_argument("--decay_res", type=float, nargs="+")
    parser.add_argument("--decay_spin", type=float, nargs="+")

    # default physical parameters
    parser.add_argument("--coupling", type=float, default=1)
    parser.add_argument("--splitting", type=float, default=0)
    parser.add_argument("--max_time", type=float, default=10)
    parser.add_argument("--time_points", type=int, default=201)

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

    # check decay arguments
    decay_com = args.decay is not None and args.decay_res is None and args.decay_spin is None
    decay_sep = args.decay is None and args.decay_res is not None and args.decay_spin is not None
    if not (decay_com ^ decay_sep):
        parser.error("Provide only --decay, XOR both --decay_res and --decay_spin")
    if decay_com:
        args.decay_res = args.decay_spin = args.decay

    return args


if __name__ == "__main__":

    start = time.time()

    args = get_simulation_args(sys.argv)
    times = np.linspace(0, args.max_time, args.time_points)
    batch_compute_QFI_vals(
        times,
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
