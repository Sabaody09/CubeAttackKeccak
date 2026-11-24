#!/usr/bin/env bash
# Reproduce the four-scenario table (Base, Discard, Discard+Pair, Discard+Quad)
# for a given number of rounds n_r, at s = 128, exactly as in the paper
# (Tables 8/9/10). Averages mu, lambda, sigma, harvested and discarded over N
# trials and reports the speedup over Base.
#
#   bash scripts/tables.sh <n_r> [N] [base_seed]
#     n_r        number of rounds (3, 4 or 5)
#     N          trials to average            (default 3; the paper used 20 / 20 / 10)
#     base_seed  trial t uses seed base+t     (default 20261120 / 20261124 / 20261119)
#
# Needs ./cube_attack_s128 (make cube_attack_s128). Six cells are run per trial:
# the four scenarios plus the UNFILTERED pair and quad runs (qd=0), which supply the
# true strategy mu and lambda (a filtered run only sees the cubes that passed the
# filter). cost = (s+1)*harvested + discarded = 129*H + D cube-sums; speedup =
# cost(Base)/cost(row). Only counts are used, never wall-clock.
set -u
cd "$(dirname "$0")/.."
BIN=./cube_attack_s128
[ -x "$BIN" ] || { echo "build first: make cube_attack_s128" >&2; exit 2; }

NR=${1:?usage: tables.sh <n_r> [N] [base_seed]}
N=${2:-3}
case "$NR" in 3) DEF=20261120;; 4) DEF=20261124;; 5) DEF=20261119;; *) DEF=20260700;; esac
BASE=${3:-$DEF}
NLEN=1216                       # IV bits = 1344 - s = 1216 for s = 128
CAP=172800                      # per-run wall-clock cap (s)
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT

[ "$NR" = 5 ] && echo "note: n_r=5 Base harvests ~36 cubes at ~minutes each; a full trial is hours." >&2

run() {  # <qd> <cube_src> <tag> <seed>
  local qd=$1 src=$2 tag=$3 seed=$4
  echo ">>> $(date '+%H:%M:%S') n_r=$NR $tag (qd=$qd cube_src=$src) seed=$seed" >&2
  "$BIN" "$NR" "$NLEN" "$seed" 50000000 "$CAP" "$qd" 128 100000000 "$src" 0 2>/dev/null \
    | awk -v tag="$tag" '/^\[params\]/{
        for(i=1;i<=NF;i++){split($i,a,"="); f[a[1]]=a[2]}
        print tag, f["mu"], f["lambda"], f["sigma"], f["harvested"], f["discarded"] }'
}

for t in $(seq 0 $((N-1))); do
  s=$((BASE + t))
  run 0 0 base       "$s" >> "$TMP"
  run 1 0 discard    "$s" >> "$TMP"
  run 1 1 pair       "$s" >> "$TMP"
  run 1 2 quad       "$s" >> "$TMP"
  run 0 1 pair_u     "$s" >> "$TMP"       # unfiltered pair  -> true pair mu, lambda
  run 0 2 quad_u     "$s" >> "$TMP"       # unfiltered quad  -> true quad mu, lambda
done

awk -v NR_="$NR" -v N="$N" '
  { n[$1]++; mu[$1]+=$2; la[$1]+=$3; sg[$1]+=$4; H[$1]+=$5; D[$1]+=$6 }
  END{
    for(k in n){ mu[k]/=n[k]; la[k]/=n[k]; sg[k]/=n[k]; H[k]/=n[k]; D[k]/=n[k] }
    cost_base = 129*H["base"]
    printf "\n=== n_r = %d, s = 128, c = 256, rate r = 1344, rank 128, N = %d trials ===\n", NR_, N
    printf "%-16s %8s %8s %10s %11s %11s %9s\n","Scenario","mu","lambda","sigma","harvested","discarded","speedup"
    row("Base",            mu["base"],  la["base"],  sg["base"],    H["base"],    D["base"],    cost_base)
    row("Discard",         mu["base"],  la["base"],  sg["discard"], H["discard"], D["discard"], cost_base)
    row("Discard + Pair",  mu["pair_u"],la["pair_u"],sg["pair"],    H["pair"],    D["pair"],    cost_base)
    row("Discard + Quad",  mu["quad_u"],la["quad_u"],sg["quad"],    H["quad"],    D["quad"],    cost_base)
    print  "mu, lambda from the unfiltered run of the same strategy; cost = 129*harvested + discarded"
  }
  function row(name,mu,la,sg,h,d,cb,   cost){ cost=129*h+d;
    printf "%-16s %8.4f %8.2f %10.0f %11.0f %11.0f %8s\n", name, mu, la, sg, h, d,
           (name=="Base") ? "-" : sprintf("%.1fx", cb/cost) }
' "$TMP"
