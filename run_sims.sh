#!/usr/bin/env zsh

states=$(for i in $(seq 1 20); do echo -n "dicke-$i "; done)
decay=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)

for num_spins in {5..20}; do
  echo -------------------------
  echo $num_spins
  echo -------------------------
  for state in ghz x-polarized $(for i in {1..$num_spins}; do echo -n "dicke-$i "; done); do
    python3 collect_data.py --num_spins $num_spins --state_keys $state --decay $decay
    python3 collect_data.py --num_spins $num_spins --state_keys $state --decay $decay --dephasing
  done
done