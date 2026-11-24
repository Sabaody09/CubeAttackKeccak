#!/usr/bin/env bash
# Run the attack for larger secret sizes s = 256 and s = 512 (no fixed seeds: a
# single demonstration each). Same cipher (rate 1344, capacity 256); only s and
# hence the number of equations to collect change. The IV shrinks to 1344 - s bits.
# The cost per harvested cube grows from 129 to s+1 cube-sums.
#
#   bash scripts/attack_s256_s512.sh [n_r] [cube_src] [seed]
#     n_r        rounds              (default 3; use 4 or 5 for more rounds)
#     cube_src   0 random 1 pair 2 quad   (default 1 = pairs, the efficient choice)
#     seed       PRNG seed           (default: current time)
#
# Needs cube_attack_s256 and cube_attack_s512 (make).
set -u
cd "$(dirname "$0")/.."
NR=${1:-3}; SRC=${2:-1}; SEED=${3:-$(date +%s)}

for s in 256 512; do
  BIN=./cube_attack_s$s
  [ -x "$BIN" ] || { echo "build first: make cube_attack_s$s" >&2; exit 2; }
  nlen=$((1344 - s))
  echo "================ s = $s, n_r = $NR, cube_src = $SRC, seed = $SEED ================"
  "$BIN" "$NR" "$nlen" "$SEED" 50000000 172800 1 "$s" 100000000 "$SRC" 0 2>&1 \
    | grep -E '^\[config|^\[preprocess|^\[params|^\[solve|^RESULT'
  echo
done
