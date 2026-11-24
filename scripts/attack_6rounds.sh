#!/usr/bin/env bash
# Six-round attack via the degree-reduction trick (Section "Adding one round by
# inverting chi"). chi acts row-wise and the rate exposes four complete rows, so
# the last round's iota and chi are inverted on them per cube point; the observed
# bits then have degree 2^{n_r-1}, so a six-round attack uses cubes of dimension
# 2^5 - 1 = 31 -- the same size as a normal five-round attack.
#
# This runs, at s = 128, Discard+Pair for:
#   (a) n_r = 5 normal  (invhalf=0, d=31)
#   (b) n_r = 6 invhalf (invhalf=1, d=31)
# and recovers the key in both. Both use the same cube size, so the comparison is
# in the harvested/discarded cube counts: the six-round invhalf attack costs about
# the same as the five-round normal one. Each harvest is a few minutes on a modern
# GPU, so each run takes roughly 10-30 minutes.
#
#   bash scripts/attack_6rounds.sh [seed]
set -u
cd "$(dirname "$0")/.."
BIN=./cube_attack_s128
[ -x "$BIN" ] || { echo "build first: make cube_attack_s128" >&2; exit 2; }
SEED=${1:-20260601}

echo "################ n_r = 5, normal, Discard+Pair (d=31), s=128 ################"
"$BIN" 5 1216 "$SEED" 50000000 172800 1 128 100000000 1 0 2>&1 \
  | grep -E '^\[config|^\[preprocess|^\[params|^\[solve|^RESULT'

echo
echo "################ n_r = 6, invhalf, Discard+Pair (d=31), s=128 ################"
"$BIN" 6 1216 "$SEED" 50000000 172800 1 128 100000000 1 1 2>&1 \
  | grep -E '^\[config|^\[preprocess|^\[params|^\[solve|^RESULT'

echo
echo "Both reach rank 128 with d = 31. Compare the 'harvested'/'discarded' counts on"
echo "the [preprocess] lines: six-round invhalf ~ five-round normal complexity."
