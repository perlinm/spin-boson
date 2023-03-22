#!/usr/bin/env python3
import os

job_dir = "jobs"
job_name = "test"
base_name = os.path.join(job_dir, job_name)

# num_spins = [2, 3, 4, 5]
# decay = [0.25 0.50 1 2 3 4 5]
num_spins = [2]
decay = [0.25]
state_keys = ["ghz"]

num_spins_str = " ".join(map(str, num_spins))
decay_str = " ".join(map(str, decay))
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
--decay {decay_str} \
--state_keys {state_keys_str}
"""

os.makedirs(job_dir, exist_ok=True)
with open(f"{base_name}.sh", "w") as file:
    file.write(log_text)
