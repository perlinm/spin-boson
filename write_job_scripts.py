#!/usr/bin/env python3
"""
Contents: Script to generate slurm jobs.
Author: Michael A. Perlin (2023)
"""
import os
import sys

job_dir = "jobs"

num_spin_vals = list(range(2, 21))
decay_res_vals = [0.2 * kk for kk in range(1, 16)]
state_keys = ["dicke-1", "dicke-2", "ghz", "x-polarized"]

decay_spin = float(sys.argv[1])
decay_spin_vals = [decay_spin]

num_spins_str = " ".join(map(str, num_spin_vals))
decay_res_str = " ".join(map(str, decay_res_vals))
decay_spin_str = " ".join(map(str, decay_spin_vals))
state_keys_str = " ".join(state_keys)

job_name = f"qfi_N{max(num_spin_vals)}_g{decay_spin:.2f}"
base_name = os.path.join(job_dir, job_name)

log_text = f"""#!/bin/sh
#SBATCH --partition=knlall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --job-name={job_name}
#SBATCH --output={base_name}.out
#SBATCH --error={base_name}.err
#SBATCH --time=24:00:00

python3 collect_data.py \
--num_spins {num_spins_str} \
--decay_res {decay_res_str} \
--decay_spin {decay_spin_str} \
--state_keys {state_keys_str}
"""

os.makedirs(job_dir, exist_ok=True)
with open(f"{base_name}.sh", "w") as file:
    file.write(log_text)
