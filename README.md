# GPU cube attack on round-reduced Keccak-MAC

This repository contains the code for the paper

> M. Calderini, M. Frighetto, C. Mascia. **Cube-like Attacks on Keccak: refined
> complexity analysis and practical enhancements**.

It implements a cube attack on Keccak-MAC instantiated with Keccak-f[1600] reduced
to *r* rounds. The secret key has *c* = 128 or 256 bits. The code is parallelized on
an NVIDIA GPU. It includes two speed-ups for the offline phase: the
**quick-discard** filter and two **cube-selection strategies** (pairs and
quadruplets).

## Contents

```
src/keccak.{c,h}    reference Keccak-f permutation, any number of rounds
src/mac.{c,h}       Keccak-MAC: key || nonce -> r rounds of Keccak-f[1600]
src/cube.{c,h}      reference (CPU) cube-sum and affine superpoly reconstruction
gpu/kf1600.cuh      Keccak-f[1600] round function shared by host and device
gpu/cube_attack.cu  GPU attack driver: offline, online and linear-algebra phases,
                    quick discard and cube selection
Makefile
```

## Requirements

- An NVIDIA GPU with CUDA support. All results in the paper were obtained on an
  NVIDIA A100-SXM4-80GB.
- CUDA toolkit 12.x (tested with 12.8) and a C compiler (tested with gcc 13.3).
- Linux with glibc. The secret key and the cube sequence are drawn with the C
  library `rand()`, so a given seed reproduces the exact same run only with
  glibc.

## Build

```sh
make                    # builds cube_attack_c128 and cube_attack_c256
make CUDA_ARCH=sm_90    # for a different GPU generation (default: sm_80, A100)
```

The key size is a compile-time constant, so there is one binary per key size. Any
multiple of 64 is supported; for example, a 512-bit key:

```sh
make keccak.o mac.o cube.o
nvcc -O3 -arch=sm_80 -Isrc -Igpu -DKLEN_BITS=512 gpu/cube_attack.cu keccak.o mac.o cube.o -o cube_attack_c512
```

## Usage

```
./cube_attack_c128 [rounds] [nlen] [seed] [max_cubes] [max_sec] [qd] [rank_target] [stall] [cube_src]
./cube_attack_c256  (same arguments)
```

| Argument      | Meaning                                                                 | Default     |
|---------------|-------------------------------------------------------------------------|-------------|
| `rounds`      | number of Keccak-f rounds *r*; the cube dimension is *d* = 2^r − 1 (*k* in the paper) | 5 |
| `nlen`        | number of public (nonce) bits available as cube variables               | 1600 − *c*  |
| `seed`        | PRNG seed; determines the secret key and the sequence of cubes          | 20260722    |
| `max_cubes`   | maximum number of cubes tried in the offline phase                      | 200000      |
| `max_sec`     | wall-clock budget of the offline phase, in seconds                      | unlimited   |
| `qd`          | quick discard: `0` off, `1` on                                          | 1           |
| `rank_target` | stop the offline phase at this rank. Below *c*, the missing bits are left to brute force | *c* |
| `stall`       | stop after this many consecutive harvested cubes bring no new rank      | 100000      |
| `cube_src`    | cube selection: `0` random, `1` pairs, `2` quadruplets                  | 0           |

The four scenarios compared in the paper correspond to:

| Scenario       | `qd` | `cube_src` |
|----------------|------|------------|
| Base           | 0    | 0          |
| Discard        | 1    | 0          |
| Discard + Pair | 1    | 1          |
| Discard + Quad | 1    | 2          |

### Examples

```sh
# 3 rounds, 128-bit key, all four scenarios (about 30 s for Base, seconds for the others)
./cube_attack_c128 3 1472 1 200000 1e12 0 128 100000 0    # Base
./cube_attack_c128 3 1472 1 200000 1e12 1 128 100000 0    # Discard
./cube_attack_c128 3 1472 1 200000 1e12 1 128 100000 1    # Discard + Pair
./cube_attack_c128 3 1472 1 200000 1e12 1 128 100000 2    # Discard + Quad

# 5 rounds, 256-bit key, quadruplets with quick discard
./cube_attack_c256 5 1344 1 200000 1e12 1 256 100000 2
```

Use a different `seed` for each independent trial. The paper's parameter values are
averages over several trials of the same configuration.

## What the attack does

The key occupies state bits 0..*c*−1 and the nonce occupies bits *c*..*c*+`nlen`−1.
All other bits are zero. A Keccak-f round has degree 2, so after *r* rounds every
output bit has degree at most 2^r. With a cube of dimension *d* = 2^r − 1, every
superpoly is therefore **affine** in the key. No linearity test is needed.

1. **Offline phase.** For a candidate cube, the driver computes *c* + 1
   full-state cube-sums: one with key = 0 and one for each unit key vector
   *e_j*. This is called a *harvest*. It gives the affine superpolys of all 1600
   output bits at once. The non-constant superpolys that are linearly independent
   of those already collected are kept. This repeats until the rank reaches *c*
   (or `rank_target`).
   - *Quick discard* (`qd=1`): before a harvest, only the key = 0 cube-sum is
     computed, which costs 1 cube-sum instead of *c* + 1. Its output gives the
     constant terms of the 1600 superpolys. If they are all zero, the cube is
     discarded.
   - *Pairs* (`cube_src=1`): cube variables are placed in pairs that are
     multiplied together in the χ step of round 1.
   - *Quadruplets* (`cube_src=2`): two such pairs are chosen so that their
     products are multiplied again in the χ step of round 2.
2. **Online phase.** One single-bit cube-sum under the secret key for each
   superpoly kept. Each one gives a linear equation in the key bits.
3. **Linear algebra.** Gaussian elimination over GF(2) recovers the key, or the
   key up to *c* − rank free bits.

### Parallelization

Each cube-sum (2^d evaluations of the reduced permutation) is one GPU task, and
the *c* + 1 tasks of a harvest are launched in a single kernel (one task per
`blockIdx.y`). Within a task, 256-thread blocks cover the 2^d cube points with a
grid-stride loop. Each thread XORs the 1600-bit outputs of its points. The
partial sums are reduced in shared memory and combined across blocks with
`atomicXor`.

## Output

Each harvested cube prints one line with the number of non-constant superpolys it
produced (`nc`), the rank gained and the total rank. The run ends with:

```
[preprocess/GPU] ... rank R/c from N cubes (D quick-discarded, H harvested) in T s
[params] tried=N discarded=D harvested=H useful=U | mu=... mu_h=... lambda=... lambda_h=... sigma=... (rank R)
[online/GPU] R equations in T s; exactness invariant: OK
[solve] rank=R/c in T s (linear algebra)
[timing] offline=... s | online=... s | linear-algebra=... s | total=... s
RESULT: FULL c-bit KEY RECOVERED (GPU, f[1600]) -- matches secret [OK]
```

The `[params]` line gives the attack parameters defined in the paper, measured on
this run:

| Parameter  | Definition                                                                 |
|------------|----------------------------------------------------------------------------|
| `mu`       | μ = useful cubes / cubes tried. A cube is useful if it yields at least one non-constant superpoly |
| `mu_h`     | μ_h = harvested cubes / cubes tried (1 without quick discard)               |
| `lambda`   | λ = non-constant superpolys per useful cube                                 |
| `lambda_h` | λ_h = non-constant superpolys per harvested cube                           |
| `sigma`    | σ = non-constant superpolys collected until the final rank R was reached    |

The cost of the offline phase, which does not depend on the hardware, is
((*c* + 1) · H + D) · 2^d Keccak calls.

## Built-in correctness checks

Every run checks that:

- the GPU cube-sum matches the C reference implementation (`src/cube.c`) on
  16 random small cubes;
- each quadruplet's product *abcd* is present in the 2-round output (100 random
  quadruplets, `cube_src=2` only);
- every collected equation holds for the secret key (exactness invariant);
- the recovered key equals the secret key, when the rank is full.

The exit code is 0 only if all checks pass.

## Expected running times (A100-SXM4-80GB)

| Rounds | *d* | One harvest, *c* = 128 | One harvest, *c* = 256 | Quick-discard probe |
|--------|-----|------------------------|------------------------|---------------------|
| 3      | 7   | ~ 2 ms                 | ~ 3.5 ms               | < 1 ms              |
| 4      | 15  | ~ 5 ms                 | ~ 10 ms                | < 1 ms              |
| 5      | 31  | ~ 80 s                 | ~ 160 s                | ~ 0.6 s             |

At 3 and 4 rounds the times are dominated by kernel launches rather than
computation. A full run takes seconds to a few minutes, the base scenario being the
slowest. At 5 rounds a full run takes minutes with pairs or quadruplets, and from
tens of minutes to a few hours in the base and discard scenarios. Random cubes
need many more harvests, and that number varies a lot from seed to seed.
