#!/usr/bin/env python3
import os
from typing import Sequence

import numpy as np

import methods


data_dir = "./data"


def get_data_dir(state_key: str) -> str:
    return os.path.join(data_dir, state_key)


def get_file_path(prefix: str, state_key: str, num_spins: int, kk: int, gg: int) -> str:
    return os.path.join(get_data_dir(state_key), f"{prefix}_N{num_spins}_k{kk}_g{gg}.txt")


eV = 27.2114
autime = 2.41888e-17
max_time = (100.0e-15) / autime / eV
times = np.linspace(0, max_time, 101)

splitting = 0
coupling = 0.04

spin_num_vals = list(range(2, 7))

# decay rates on resonator (kappa) and qubits (gamma)
kappa_vals = np.array([0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24])
gamma_vals = np.array([0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24])


get_initial_state = {
    "ghz": methods.get_ghz_state,
    "dicke_1": lambda num_spins: methods.get_dicke_state(num_spins, 1),
    "dicke_2": lambda num_spins: methods.get_dicke_state(num_spins, 2),
}


def compute_fisher_vals(
    times: np.ndarray,
    spin_num_vals: Sequence[int],
    kappa_vals: np.ndarray,
    gamma_vals: np.ndarray,
    splitting: float,
    coupling: float,
    state_key: str,
    status_updates: bool = False,
) -> None:
    os.makedirs(get_data_dir(state_key), exist_ok=True)

    for num_spins in spin_num_vals:
        for kk, kappa in enumerate(kappa_vals):
            for gg, gamma in enumerate(gamma_vals):
                if status_updates:
                    print(num_spins, kk, gg)
                fisher_vals, scaled_fisher_vals = methods.get_fisher_vals(
                    times,
                    num_spins,
                    splitting,
                    coupling,
                    kappa,
                    gamma,
                    get_initial_state[state_key](num_spins),
                )
                vals_file = get_file_path("fisher", state_key, num_spins, kk, gg)
                scaled_vals_file = get_file_path("scaled_fisher", state_key, num_spins, kk, gg)
                np.savetxt(vals_file, fisher_vals)
                np.savetxt(scaled_vals_file, scaled_fisher_vals)


if __name__ == "__main__":

    for state_key in get_initial_state.keys():
        print("-" * 80)
        print(state_key)
        print("-" * 80)
        compute_fisher_vals(
            times,
            spin_num_vals,
            kappa_vals,
            gamma_vals,
            splitting,
            coupling,
            state_key,
            status_updates=True,
        )
