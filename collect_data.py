#!/usr/bin/env python3
"""Script to simulate a spin-boson system."""
import argparse
import itertools
import multiprocessing
import os
import re
import sys
import time
from typing import Callable, Sequence

import numpy as np

import methods


def get_data_dir(data_dir: str, state_key: str) -> str:
    return os.path.join(data_dir, state_key)


def get_file_path(
    data_dir: str,
    prefix: str,
    state_key: str,
    num_spins: int,
    decay_res: float,
    decay_spin: float,
    dephasing: bool,
) -> str:
    base_tag = f"{state_key}_N{num_spins}_k{decay_res:.2f}_g{decay_spin:.2f}"
    if dephasing:
        base_tag += "_z"
    return os.path.join(data_dir, f"{prefix}_{base_tag}.txt")


def state_constructor(state_key: str) -> Callable[[int], np.ndarray]:
    if state_key == "x-polarized":
        return methods.get_state_X
    if state_key == "ghz":
        return methods.get_ghz_state
    if re.match("dicke-[0-9]+$", state_key):
        excitations = int(state_key.strip("dicke-"))
        return lambda num_spins: methods.get_dicke_state(
            num_spins, excitations, boson_dim=excitations + 1
        )
    raise ValueError(f"initial state not recognized: {state_key}")


def compute_QFI_vals(
    times: np.ndarray,
    num_spins: int,
    decay_res: float,
    decay_spin: float,
    dephasing: bool,
    spin_splitting: float,
    boson_splitting: float,
    coupling: float,
    initial_state: np.ndarray,
    file_QFI: str,
    status_update: bool = False,
) -> None:
    if status_update:
        print(os.path.relpath(file_QFI))
        sys.stdout.flush()
    vals_QFI = methods.get_QFI_vals(
        times,
        num_spins,
        spin_splitting,
        boson_splitting,
        coupling,
        decay_res,
        decay_spin,
        dephasing,
        initial_state,
    )
    sim_times = times[: len(vals_QFI)]

    os.makedirs(os.path.dirname(file_QFI), exist_ok=True)
    np.savetxt(file_QFI, np.array([sim_times, vals_QFI]).T, header="time, QFI")


def batch_compute_QFI_vals(
    times: np.ndarray,
    num_spin_vals: Sequence[int],
    decay_res_vals: Sequence[float],
    decay_spin_vals: Sequence[float],
    dephasing: bool,
    spin_splitting: float,
    boson_splitting: float,
    coupling: float,
    state_keys: Sequence[str],
    data_dir: str,
    num_jobs: int = 1,
    recompute: bool = False,
    status_update: bool = False,
) -> None:
    if num_jobs > 1:
        processes = num_jobs if num_jobs >= 0 else multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=processes)
        results = []

    for num_spins, decay_res, decay_spin, state_key in itertools.product(
        num_spin_vals, decay_res_vals, decay_spin_vals, state_keys
    ):
        if re.match("dicke-[0-9]+$", state_key) and int(state_key.strip("dicke-")) > num_spins:
            continue

        args = (state_key, num_spins, decay_res, decay_spin, dephasing)
        file_QFI = get_file_path(data_dir, "qfi", *args)

        if not os.path.isfile(file_QFI) or recompute:
            initial_state = state_constructor(state_key)(num_spins)
            job_args = (
                times,
                num_spins,
                decay_res,
                decay_spin,
                dephasing,
                spin_splitting,
                boson_splitting,
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
        description="Compute QFI.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--state_keys", type=str, nargs="+", required=True)
    parser.add_argument("--num_spins", type=int, nargs="+", required=True)

    parser.add_argument("--decay", type=float, nargs="+")
    parser.add_argument("--decay_res", type=float, nargs="+")
    parser.add_argument("--decay_spin", type=float, nargs="+")
    parser.add_argument("--dephasing", action="store_true")

    # default physical parameters
    parser.add_argument("--coupling", type=float, default=1)
    parser.add_argument("--spin_splitting", type=float, default=0)
    parser.add_argument("--boson_splitting", type=float, default=0)
    parser.add_argument("--max_time", type=float, default=10)
    parser.add_argument("--time_points", type=int, default=201)

    # default directories
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    default_data_dir = os.path.join(script_dir, "data")
    parser.add_argument("--data_dir", type=str, default=default_data_dir)

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
        args.dephasing,
        args.spin_splitting,
        args.boson_splitting,
        args.coupling,
        args.state_keys,
        args.data_dir,
        args.num_jobs,
        args.recompute,
        status_update=True,
    )

    print("DONE")
    print(time.time() - start)
