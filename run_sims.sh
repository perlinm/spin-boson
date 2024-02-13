#!/usr/bin/env zsh

decay=(0 0.2 0.4 0.6 0.8 1.0)

for num_spins in {0..20}; do
  echo -------------------------
  echo $num_spins
  echo -------------------------
  for state in ghz x-polarized $(for ii in {0..15}; do echo "dicke-$ii"; done); do
    python3 collect_data.py --num_spins $num_spins --state_keys $state --decay $decay
    python3 collect_data.py --num_spins $num_spins --state_keys $state --decay $decay --dephasing
  done
done
