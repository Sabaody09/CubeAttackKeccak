# GPU cube attack on a round-reduced Keccak stream cipher

Code for the paper

> M. Calderini, M. Frighetto, C. Mascia. **Cube-like Attacks on Keccak: refined
> complexity analysis and practical enhancements**.

It implements a GPU-parallel cube attack on a stream cipher built from the
round-reduced Keccak-f[1600] permutation, and the two offline-phase optimizations
of the paper — the **quick-discard** filter and the **pair / quadruplet**
cube-selection strategies — plus the **degree-reduction** trick that inverts the
last half-round to gain one round.

## Scenario

Keccak-f[1600] with bitrate `r = 1344` and capacity `c = 256` (both fixed, as in
SHAKE128). The keystream is the whole rate after `n_r` rounds:

```
keystream = first r = 1344 bits of  Keccak-f[1600]^{n_r}( key || IV || 0^c )

   state bits      0 .. s-1      s   secret key bits   (the attack target)
   state bits      s .. 1343     IV  public bits       (cube variables live here)
   state bits   1344 .. 1599     capacity, fixed to 0, never observed
```

The attacker chooses the IV and observes the whole rate (chosen-IV, known
keystream). The secret size **s** is a compile-time constant (128, 256 or 512);
the round count **n_r**, the cube strategy and the optimizations are run-time
arguments. The symbols match the paper: `n_r` rounds, `s` secret bits, cube
dimension `k = 2^{n_r} - 1`, and the parameters `μ`, `λ`, `σ`.

## Contents

```
src/keccak.{c,h}      reference Keccak-f[1600] permutation (any n_r), round-constant
                      inverse of the last half-round (keccak_inv_last_half)
src/keystream.{c,h}   keystream oracle: (key || IV || 0) -> n_r rounds -> output bits
src/cube.{c,h}        reference (CPU) cube-sum, used to validate the GPU
gpu/kf1600.cuh        Keccak-f[1600] round function + half-round inverse, host & device
gpu/cube_attack.cu    the attack: offline, online and linear-algebra phases, quick
                      discard, pair/quadruplet selection, degree reduction (invhalf)
tests/keccak_kat.c    SHA3-256 known-answer tests
tests/halfround_test.c  the chi/iota inversion and the degree drop (2^{n_r} -> 2^{n_r-1})
scripts/              reproduce the tables (tables.sh), the s=256/512 runs, the
                      six-round attack; seeds.txt lists the published seeds
Makefile
```

## Requirements

- An NVIDIA GPU with CUDA. The published results were obtained on an A100-SXM4-80GB
  and reproduced on an NVIDIA GB10.
- CUDA toolkit (`nvcc`, tested 12.8 and 13.0) and a C compiler (tested gcc 13.3).
- Linux with glibc: the secret key and the cube sequence come from glibc `rand()`,
  so a seed reproduces a run exactly only on glibc (the statistics do not depend on
  the libc).

## Build

```sh
make                      # cube_attack_s128, cube_attack_s256, cube_attack_s512
make CUDA_ARCH=sm_121     # pick the GPU arch (default sm_80 = A100; sm_121 = GB10, sm_90 = H100)
make check                # CPU self-tests (no GPU): SHA3 KATs + half-round inversion
```

One binary per secret size, because `s` sets the width of the key vectors and of
the GF(2) system. `s` must be a multiple of 64.

## Usage

```
./cube_attack_s<s> [n_r] [nlen] [seed] [max_cubes] [max_sec] [qd] [rank_target] [stall] [cube_src] [invhalf]
```

| Argument      | Meaning                                                                      | Default      |
|---------------|------------------------------------------------------------------------------|--------------|
| `n_r`         | number of Keccak-f rounds; cube dimension `k = 2^{n_r} - 1` (or `2^{n_r-1}-1` with `invhalf`) | 5 |
| `nlen`        | number of public IV bits (cube variables); `s + nlen ≤ 1344`                 | `1344 - s`   |
| `seed`        | PRNG seed; fixes the secret key and the whole cube sequence                  | 20260722     |
| `max_cubes`   | cube budget for the offline phase                                            | 200000       |
| `max_sec`     | wall-clock budget of the offline phase, in seconds                           | unlimited    |
| `qd`          | quick discard: `0` off, `1` on                                               | 1            |
| `rank_target` | stop the offline phase at this rank; below `s` the missing bits are brute-forced | `s`      |
| `stall`       | stop after this many consecutive harvested cubes bring no new rank           | 100000       |
| `cube_src`    | cube selection: `0` random, `1` pairs, `2` quadruplets                       | 0            |
| `invhalf`     | degree reduction: `0` off, `1` invert the last half-round (see below)        | 0            |

The four scenarios of the paper:

| Scenario        | `qd` | `cube_src` |
|-----------------|------|------------|
| Base            | 0    | 0          |
| Discard         | 1    | 0          |
| Discard + Pair  | 1    | 1          |
| Discard + Quad  | 1    | 2          |

### Examples

```sh
# s=128, 3 rounds, Discard+Pair: recovers the key in well under a second
./cube_attack_s128 3 1216 20261120 5000000 300 1 128 100000000 1 0

# s=256, 4 rounds, Discard+Quad
./cube_attack_s256 4 1088 12345 5000000 3600 1 256 100000000 2 0

# s=128, 6 rounds via the degree-reduction trick (invhalf=1 -> cube dimension 31)
./cube_attack_s128 6 1216 20260601 50000000 172800 1 128 100000000 1 1
```

## What the attack does

A Keccak round has algebraic degree 2, so after `n_r` rounds every output bit has
degree ≤ `2^{n_r}`. With a cube of dimension `k = 2^{n_r} - 1` every superpoly is
therefore **affine** in the key — no linearity test is needed.

1. **Offline phase.** For a candidate cube the driver computes `s + 1` full-state
   cube-sums: one with `key = 0` and one per unit key vector `e_j` (a *harvest*).
   This gives the affine superpolys of all observable output bits at once; the
   non-constant ones that are linearly independent of those already collected are
   kept, until the rank reaches `s` (or `rank_target`).
   - **Quick discard** (`qd=1`): before harvesting, compute only the `key = 0`
     cube-sum (1 cube-sum instead of `s+1`). If every observable constant term is 0
     the cube is discarded.
   - **Pairs** (`cube_src=1`): IV variables are placed in pairs multiplied together
     in the round-1 χ.
   - **Quadruplets** (`cube_src=2`): two such pairs whose products are multiplied
     again in the round-2 χ.
2. **Online phase.** One single-bit cube-sum under the secret key per kept
   superpoly; each gives a linear equation in the key bits.
3. **Linear algebra.** Gaussian elimination over GF(2) recovers the key (or the key
   up to `s − rank` bits, left to brute force).

The `[params]` line reports the paper's parameters measured on the run:

| Parameter  | Definition                                                                 |
|------------|----------------------------------------------------------------------------|
| `mu` (μ)   | useful cubes / cubes tried (a cube is *useful* if it yields ≥ 1 non-constant superpoly) |
| `lambda` (λ) | non-constant superpolys per useful cube                                   |
| `sigma` (σ) | non-constant superpolys collected until the final rank was reached         |

The hardware-independent offline cost is `((s + 1)·H + D)·2^k` Keccak calls, with
`H` harvested and `D` discarded cubes (Theorem 3 of the paper).

### Adding one round by inverting χ (degree reduction, `invhalf=1`)

Because χ acts row-wise and the rate exposes four complete rows (lanes 0..19), the
last round's ι and χ can be inverted on those rows from the observed keystream,
recovering the state just before the last χ. That state is a linear image of the
state after `n_r − 1` rounds, so the observed bits have degree `2^{n_r-1}` instead
of `2^{n_r}`: cubes of dimension `2^{n_r-1} − 1` suffice. A **six-round** attack
then uses `k = 2^5 − 1 = 31`, the same cube size as a normal five-round attack, and
costs essentially the same (`scripts/attack_6rounds.sh` runs both and compares).
The inversion is a handful of bitwise operations per cube point, negligible next to
the `2^k` permutations. In this mode only lanes 0..19 (1280 bits) are observable.
The trick is generic in `n_r`.

## Reproducing the paper's tables

```sh
bash scripts/tables.sh 3 20 20261120      # Table 8  (n_r = 3, s = 128)
bash scripts/tables.sh 4 20 20261124      # Table 9  (n_r = 4, s = 128)
bash scripts/tables.sh 5 10 20261119      # Table 10 (n_r = 5, s = 128; Base is hours)
```

`tables.sh <n_r> [N] [base_seed]` runs the four scenarios over `N` trials
(trial `t` uses `base_seed + t`), plus the unfiltered pair/quad runs that give the
true strategy `μ`/`λ`, and prints the averaged table with the speedup over Base.
See `scripts/seeds.txt`. The numbers match the paper up to trial noise; a smaller
`N` (default 3) reproduces them approximately in a fraction of the time.

Larger secret sizes and the six-round attack:

```sh
bash scripts/attack_s256_s512.sh 3 1       # recover a 256- and a 512-bit key (no seeds)
bash scripts/attack_6rounds.sh             # five-round normal vs six-round invhalf
```

## Built-in correctness checks

Every run checks that the GPU cube-sum matches the C reference (`src/cube.c`,
`src/keccak.c`) on random points — with the matching inversion in `invhalf` mode;
that each quadruplet's product `abcd` is present after two rounds (`cube_src=2`);
that every collected equation holds for the secret key (exactness invariant); and
that the recovered key equals the secret when the rank is full. The exit code is 0
only if all checks pass. `make check` additionally verifies the SHA3 KATs and the
degree-reduction inversion on the CPU.

## Parallelization

Each cube-sum (`2^k` evaluations of the round-reduced permutation) is one GPU task;
the `s + 1` tasks of a harvest run together (one per `blockIdx.y`). Within a task,
256-thread blocks cover the cube points with a grid-stride loop, each thread XORs
the 1600-bit outputs of its points (inverting the last half-round first when
`invhalf=1`), and the partial sums are reduced in shared memory and combined across
blocks with `atomicXor`. Cube generation and the GF(2) solve run on the CPU.

## Citation

```bibtex
@misc{CalderiniFrighettoMascia,
  title  = {Cube-like Attacks on Keccak: refined complexity analysis and practical enhancements},
  author = {Calderini, Marco and Frighetto, Mirko and Mascia, Carla},
  year   = {2026}
}
```

## License

[LICENSE]
