#!/usr/bin/env python3
import os

job_dir = "jobs"
job_name = "test"
base_name = os.path.join(job_dir, job_name)

# num_spins = [2, 3, 4, 5]
# decay_res = [0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24]
# decay_spin = decay_res

num_spins = [2]
decay_res = [0.01]
decay_spin = decay_res
state_keys = ["ghz"]

num_spins_str = " ".join(map(str, num_spins))
decay_res_str = " ".join(map(str, decay_res))
decay_spin_str = " ".join(map(str, decay_spin))
state_keys_str = " ".join(state_keys)

log_text = f"""#!/bin/sh

#SBATCH --partition=knlall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name={job_name}
#SBATCH --output={base_name}.out
#SBATCH --error={base_name}.err
#SBATCH --time=00:01:00

python3 collect_data.py \
--num_spins {num_spins_str} \
--decay_res {decay_res_str} \
--decay_spin {decay_spin_str} \
--state_keys {state_keys_str}
"""

os.makedirs(job_dir, exist_ok=True)
with open(f"{base_name}.sh", "w") as file:
    file.write(log_text)
