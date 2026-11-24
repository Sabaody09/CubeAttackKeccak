/*
 * GPU-parallel cube attack on a round-reduced Keccak stream cipher (Keccak-f[1600]).
 *
 * SCENARIO (rate r = b - c = 1344, capacity c = 256, both fixed):
 *   keystream = first r = 1344 bits of  Keccak-f[1600]^{n_r}( key || IV || 0^256 )
 *     - key : s secret bits at state positions 0..s-1   (the attack target);
 *     - IV  : nlen public bits at positions s..s+nlen-1, inside the rate, so
 *             s + nlen <= 1344; the cube variables live here;
 *     - the 256 capacity bits are fixed to 0 and never observed.
 *   The attacker observes the whole rate (chosen IV, known keystream). The secret
 *   size s is fixed at compile time: -DS_BITS=128 (default), 256 or 512.
 *
 * Attack (cube dimension d = 2^{n_r} - 1, so every superpoly is affine in the key):
 *   offline  For each candidate cube, compute the cube-sum of the output state for
 *            key=0 and key=e_j (j=0..s-1): s+1 cube-sums of 2^d evaluations each
 *            ("harvest"). This yields the affine superpolys of all observable
 *            output bits at once; keep the non-constant, linearly independent ones
 *            until rank s (or rank_target) is reached.
 *   online   One single-bit cube-sum per kept superpoly under the secret key.
 *   solve    Gaussian elimination over GF(2).
 *
 * Offline speed-ups (command-line switches):
 *   qd=1        quick discard: before a harvest run only the key=0 cube-sum (1
 *               instead of s+1). Its observable bits are the constant terms of the
 *               superpolys; if all are zero the cube is skipped.
 *   cube_src=1  pairs: cube variables placed in pairs multiplied in the round-1 chi.
 *   cube_src=2  quadruplets: two pairs whose products are multiplied again in the
 *               round-2 chi.
 *   invhalf=1   degree reduction: chi acts row-wise and the rate exposes four
 *               complete rows, so the last round's iota and chi are inverted on
 *               them per cube point. The observed bits then have degree 2^{n_r-1},
 *               so cubes of dimension 2^{n_r-1}-1 suffice and an n_r-round attack
 *               costs like an (n_r-1)-round one. Observable region shrinks to lanes
 *               0..19 (1280 bits). Generic in n_r (used for the six-round attack).
 *
 * Parallelization: each cube-sum is one task; the s+1 tasks of a harvest are
 * launched together (grid.y = task). Within a task, threads take the 2^d cube
 * points with a grid-stride loop, XOR their 1600-bit outputs locally, reduce in
 * shared memory and combine blocks with atomicXor.
 *
 * After the offline phase a [params] line reports the run's attack parameters:
 * mu, mu_h, lambda, lambda_h and sigma (definitions next to the printf).
 *
 * Built-in checks:
 *   - GPU cube-sum vs the C reference (src/cube.c, src/keccak.c) on 16 random
 *     small cubes (with the matching inversion in invhalf mode);
 *   - quadruplet construction: a*b*c*d present after 2 rounds (cube_src=2 only);
 *   - online exactness invariant: every equation holds for the secret key;
 *   - recovered key == secret key (on full rank).
 *
 * Usage: ./cube_attack_s<S> [n_r] [nlen] [seed] [max_cubes] [max_sec]
 *                           [qd] [rank_target] [stall] [cube_src] [invhalf]
 *   n_r         reduced round count                        (default 5)
 *   nlen        # IV (public) bits                         (default 1344 - s)
 *   seed        PRNG seed (secret key and cube sequence)   (default 20260722)
 *   max_cubes   cube budget                                (default 200000)
 *   max_sec     offline wall-clock budget, s               (default 1e12)
 *   qd          quick discard 0=off, 1=on                  (default 1)
 *   rank_target stop the offline phase at this rank        (default s = full key)
 *   stall       stop after this many consecutive harvested cubes add no rank
 *                                                          (default 100000)
 *   cube_src    0=random, 1=pairs, 2=quadruplets            (default 0)
 *   invhalf     0=normal, 1=invert last half-round          (default 0)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include "kf1600.cuh"

extern "C" {
#include "keccak.h"
#include "keystream.h"
#include "cube.h"
}

#define CUDA_CHECK(call) do {                                              \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);      \
    }                                                                      \
} while (0)

/* ----- key size -----
 * Compile with -DS_BITS=128 or -DS_BITS=256. S must be a multiple of 64.
 * The key occupies state bits 0..S-1 (lanes 0..S/64-1). */
#ifndef S_BITS
#define S_BITS 128
#endif
#if (S_BITS % 64) != 0 || S_BITS < 64 || S_BITS > 1536
#error "S_BITS must be a multiple of 64 in [64, 1536]"
#endif
#define S    S_BITS
#define SW      (S / 64)   /* 64-bit words per key vector                      */
#define MAXTASK (S + 8)    /* >= S+1 tasks per harvest and >= S equations */

/* Degree-reduction mode (Section "Adding one round by inverting chi"): when 1,
 * every cube-sum inverts the last round's iota and chi on the four complete rows
 * y=0..3 (lanes 0..19) via inv_last_half() in kf1600.cuh, so the observed bits
 * have algebraic degree 2^(n_r-1) and cubes of dimension 2^(n_r-1)-1 suffice. The
 * observable region is then lanes 0..19 (1280 bits). Set once in main; read by the
 * two batch wrappers and passed to the kernels. Forced off during quad_selftest. */
static int g_invhalf = 0;

/* --------------------------------- kernels -------------------------------- */
/* Single-bit cube-sum per task (used online). */
__global__ void cubesum_bit_kernel(const uint64_t *base_states, const int *posarr,
        const int *darr, const int *outarr, const unsigned long long *totarr,
        int nr, int invhalf, unsigned *result) {
    int task = blockIdx.y;
    const uint64_t *bs  = base_states + (size_t)task * 25;
    const int      *pos = posarr      + (size_t)task * KF1600_DMAX;
    int d = darr[task], outb = outarr[task];
    unsigned long long total = totarr[task];
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned local = 0;
    for (unsigned long long m = idx; m < total; m += stride)
        local ^= (unsigned)eval_point64_i(bs, pos, d, m, nr, invhalf, outb);
    __shared__ unsigned sh[256];
    sh[threadIdx.x] = local; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] ^= sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicXor(&result[task], sh[0] & 1u);
}

/* Full-state cube-sum per task: XOR of the whole 1600-bit output over the cube. */
__global__ void cubesum_state_kernel(const uint64_t *base_states, const int *posarr,
        const int *darr, const unsigned long long *totarr,
        int nr, int invhalf, unsigned long long *result) {
    int task = blockIdx.y;
    const uint64_t *bs  = base_states + (size_t)task * 25;
    const int      *pos = posarr      + (size_t)task * KF1600_DMAX;
    int d = darr[task];
    unsigned long long total = totarr[task];
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    unsigned long long local[25];
    for (int L = 0; L < 25; L++) local[L] = 0ULL;
    for (unsigned long long m = idx; m < total; m += stride) {
        uint64_t A[5][5];
        for (int L = 0; L < 25; L++) A[L % 5][L / 5] = bs[L];
        for (int i = 0; i < d; i++)
            if ((m >> i) & 1ULL) { int p = pos[i]; A[(p>>6)%5][(p>>6)/5] |= (1ULL << (p & 63)); }
        permute64(A, nr);
        if (invhalf) inv_last_half(A, nr);   /* peel last iota,chi on rows 0..3 */
        for (int L = 0; L < 25; L++) local[L] ^= A[L % 5][L / 5];
    }
    __shared__ unsigned long long sh[256];
    for (int L = 0; L < 25; L++) {
        sh[threadIdx.x] = local[L]; __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s) sh[threadIdx.x] ^= sh[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) atomicXor(&result[(size_t)task * 25 + L], sh[0]);
        __syncthreads();
    }
}

struct GpuCtx { uint64_t *d_base; int *d_pos; int *d_d; int *d_out;
                unsigned long long *d_tot; unsigned *d_res; unsigned long long *d_state; };
static void ctx_init(GpuCtx *g) {
    CUDA_CHECK(cudaMalloc(&g->d_base,  (size_t)MAXTASK * 25 * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&g->d_pos,   (size_t)MAXTASK * KF1600_DMAX * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g->d_d,     MAXTASK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g->d_out,   MAXTASK * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g->d_tot,   MAXTASK * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&g->d_res,   MAXTASK * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&g->d_state, (size_t)MAXTASK * 25 * sizeof(unsigned long long)));
}
static int grid_x(unsigned long long maxtot, int TPB) {
    unsigned long long need = (maxtot + TPB - 1) / TPB;
    int bx = need > 65536ULL ? 65536 : (int)need; return bx < 1 ? 1 : bx;
}
static void run_bit_batch(GpuCtx *g, const uint64_t *base_states, const int *posarr,
        const int *darr, const int *outarr, const unsigned long long *totarr,
        int ntask, int nr, unsigned *results) {
    CUDA_CHECK(cudaMemcpy(g->d_base, base_states, (size_t)ntask*25*sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g->d_pos,  posarr, (size_t)ntask*KF1600_DMAX*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g->d_d,    darr,   ntask*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g->d_out,  outarr, ntask*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g->d_tot,  totarr, ntask*sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g->d_res, 0, ntask*sizeof(unsigned)));
    unsigned long long maxtot = 1;
    for (int t = 0; t < ntask; t++) if (totarr[t] > maxtot) maxtot = totarr[t];
    dim3 grid(grid_x(maxtot, 256), ntask);
    cubesum_bit_kernel<<<grid, 256>>>(g->d_base, g->d_pos, g->d_d, g->d_out, g->d_tot, nr, g_invhalf, g->d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(results, g->d_res, ntask*sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (int t = 0; t < ntask; t++) results[t] &= 1u;
}
static void run_state_batch(GpuCtx *g, const uint64_t *base_states, const int *posarr,
        const int *darr, const unsigned long long *totarr,
        int ntask, int nr, unsigned long long *results) {
    CUDA_CHECK(cudaMemcpy(g->d_base, base_states, (size_t)ntask*25*sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g->d_pos,  posarr, (size_t)ntask*KF1600_DMAX*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g->d_d,    darr,   ntask*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g->d_tot,  totarr, ntask*sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g->d_state, 0, (size_t)ntask*25*sizeof(unsigned long long)));
    unsigned long long maxtot = 1;
    for (int t = 0; t < ntask; t++) if (totarr[t] > maxtot) maxtot = totarr[t];
    dim3 grid(grid_x(maxtot, 256), ntask);
    cubesum_state_kernel<<<grid, 256>>>(g->d_base, g->d_pos, g->d_d, g->d_tot, nr, g_invhalf, g->d_state);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(results, g->d_state, (size_t)ntask*25*sizeof(unsigned long long), cudaMemcpyDeviceToHost));
}

/* -------------------- S-bit key vector over GF(2) (BV) ----------------- */
typedef struct { uint64_t w[SW]; } BV;
static inline void bv_zero(BV *v)              { for (int i=0;i<SW;i++) v->w[i]=0; }
static inline int  bv_iszero(const BV *v)      { for (int i=0;i<SW;i++) if(v->w[i]) return 0; return 1; }
static inline void bv_setbit(BV *v,int j)      { v->w[j>>6] |= 1ULL<<(j&63); }
static inline int  bv_testbit(const BV*v,int j){ return (int)((v->w[j>>6]>>(j&63))&1ULL); }
static inline void bv_xor(BV *d,const BV *s)   { for (int i=0;i<SW;i++) d->w[i]^=s->w[i]; }
static inline int  bv_lowbit(const BV *v)      { for (int i=0;i<SW;i++) if(v->w[i]) return i*64+__builtin_ctzll(v->w[i]); return -1; }
static inline int  bv_parity_and(const BV*a,const BV*b){ uint64_t x=0; for(int i=0;i<SW;i++) x^=a->w[i]&b->w[i]; return __builtin_popcountll(x)&1; }

static double now_sec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

/* base state: klen key bits at positions 0..klen-1, iv bits at klen.., cube
 * bits forced to 0 in the base (the kernel flips them during evaluation). */
static void build_base_state(const keccak_params *P, const uint8_t *key,
        const uint8_t *base_iv, const int *cube_idx, int d, uint64_t out_state[25]) {
    lane_t A[5][5]; memset(A, 0, sizeof A);
    for (int j = 0; j < P->klen; j++) state_set_bit(A, P->w, j, key[j]);
    for (int i = 0; i < P->nlen; i++) state_set_bit(A, P->w, P->klen + i, base_iv[i]);
    for (int i = 0; i < d; i++)       state_set_bit(A, P->w, P->klen + cube_idx[i], 0);
    for (int L = 0; L < 25; L++) out_state[L] = A[L % 5][L / 5];
}

/* One GPU single-bit cube-sum, for validation against the C reference. */
static int gpu_cube_sum(GpuCtx *g, const keccak_params *P, const uint8_t *key,
        const uint8_t *base_iv, const int *cube_idx, int d, int out_bit_pos) {
    uint64_t bs[25];
    build_base_state(P, key, base_iv, cube_idx, d, bs);
    int posbuf[KF1600_DMAX];
    for (int i = 0; i < d; i++) posbuf[i] = P->klen + cube_idx[i];
    int dd = d, oo = out_bit_pos; unsigned long long tt = 1ULL << d; unsigned r;
    run_bit_batch(g, bs, posbuf, &dd, &oo, &tt, 1, P->nr, &r);
    return (int)(r & 1);
}

/* Independent CPU reference for the degree-reduction (invhalf) cube-sum: the SAME
 * quantity the GPU computes with g_invhalf=1, but via the separate keccak.c
 * permutation and keccak_inv_last_half. out_bit_pos must lie in lanes 0..19. Used
 * only to validate the GPU on a small cube. */
static int ref_cube_sum_invhalf(const keccak_params *P, const uint8_t *key,
        const uint8_t *base_iv, const int *cube_idx, int d, int out_bit_pos) {
    uint8_t iv[1600]; memcpy(iv, base_iv, (size_t)P->nlen);
    int acc = 0; unsigned long long total = 1ULL << d;
    for (unsigned long long m = 0; m < total; m++) {
        for (int i = 0; i < d; i++) iv[cube_idx[i]] = (uint8_t)((m >> i) & 1);
        lane_t A[5][5];
        for (int x = 0; x < 5; x++) for (int y = 0; y < 5; y++) A[x][y] = 0;
        for (int j = 0; j < P->klen; j++) state_set_bit(A, P->w, j, key[j]);
        for (int i = 0; i < P->nlen; i++) state_set_bit(A, P->w, P->klen + i, iv[i]);
        keccak_permute(A, P->w, P->nr);
        keccak_inv_last_half(A, P->w, P->nr);
        acc ^= state_get_bit(A, P->w, out_bit_pos);
    }
    return acc & 1;
}

/* Harvest: full output-state cube-sums for key=0 and key=e_j (klen+1 tasks). */
static unsigned long long states[MAXTASK * 25];
static void reconstruct_harvest(GpuCtx *g, const keccak_params *P, const uint8_t *base_iv,
        const int *cube_idx, int d) {
    int klen = P->klen, ntask = klen + 1;
    static uint64_t bstates[MAXTASK * 25];
    static int posarr[MAXTASK * KF1600_DMAX];
    static int darr[MAXTASK];
    static unsigned long long totarr[MAXTASK];
    uint8_t key[S];
    for (int task = 0; task < ntask; task++) {
        memset(key, 0, klen);
        if (task > 0) key[task - 1] = 1;
        build_base_state(P, key, base_iv, cube_idx, d, &bstates[task * 25]);
        for (int i = 0; i < d; i++) posarr[task * KF1600_DMAX + i] = P->klen + cube_idx[i];
        darr[task] = d; totarr[task] = 1ULL << d;
    }
    run_state_batch(g, bstates, posarr, darr, totarr, ntask, P->nr, states);
}

/* Quick-discard pre-check: one key=0 full-state cube-sum (1 task = 2^d evals).
 * The first `obits` output bits are the constant terms of the observable
 * superpolys (the whole rate, lanes 0..20; or lanes 0..19 in invhalf mode).
 * Returns 1 iff all of them are zero (=> discard candidate), ~(klen+1)x cheaper
 * than a full harvest. obits is a multiple of 64 (1344 or 1280), so this checks
 * whole lanes only. */
static int allzero_constants(GpuCtx *g, const keccak_params *P, const uint8_t *base_iv,
        const int *cube_idx, int d, int nr, int obits) {
    uint8_t zkey[S]; memset(zkey, 0, sizeof zkey);
    uint64_t bs0[25]; build_base_state(P, zkey, base_iv, cube_idx, d, bs0);
    int pos0[KF1600_DMAX] = {0};
    for (int i = 0; i < d; i++) pos0[i] = P->klen + cube_idx[i];
    int d0 = d; unsigned long long tt0 = 1ULL << d; unsigned long long cst[25];
    run_state_batch(g, bs0, pos0, &d0, &tt0, 1, nr, cst);
    for (int L = 0; L < obits / 64; L++) if (cst[L]) return 0;
    return 1;
}

/* -------------------------- cube selection strategies ---------------------
 * Cube indices are IV-RELATIVE: state position = klen + cube_idx[i]. A
 * position p is lane L = p>>6 = x+5y, z = p&63. Any partner that falls outside
 * the iv range (e.g. into the key lanes) is rejected and redrawn. */

/* rho offset of lane (x+5y), from the RHO table in kf1600.cuh. */
static const int rho_lane[25] = { 0, 1,62,28,27, 36,44, 6,55,20,
                                  3,10,43,25,39, 41,45,15,21, 8, 18, 2,61,56,14 };

/* Strategy 1: pairs. Each odd-indexed variable is placed so that, after
 * theta->rho->pi, it lands in the same chi row, the adjacent chi column and the
 * same z-slice as its even-indexed partner, so the two are ANDed in round-1 chi.
 * Partner of source lane (x,y) is (x+1,y+1) with z shifted by
 * rho[prev]-rho[partner] (theta diffusion ignored: primary-copy model). */
static void gen_cube_pairing(int *cube_idx, int d, int klen, int nlen) {
    unsigned char used[25 * 64]; memset(used, 0, sizeof used);
    int words[KF1600_DMAX], bits[KF1600_DMAX];
    int count = 0;
    while (count < d) {
        int word, bit, pos;
        if (count % 2 == 0) {                       /* free (random) endpoint */
            pos  = klen + rand() % nlen;
            word = pos >> 6; bit = pos & 63;
            if (used[pos]) continue;
        } else {                                    /* partner that collides in round-1 chi */
            int pw = words[count - 1], pb = bits[count - 1];
            int px = pw % 5, py = pw / 5;
            word = (px + 1) % 5 + 5 * ((py + 1) % 5);
            bit  = (((pb + rho_lane[pw] - rho_lane[word]) % 64) + 64) % 64;
            pos  = word * 64 + bit;
            if (pos < klen || pos >= klen + nlen || used[pos]) {
                used[words[count - 1] * 64 + bits[count - 1]] = 0;
                count--; continue;
            }
        }
        used[pos] = 1; words[count] = word; bits[count] = bit;
        cube_idx[count] = pos - klen; count++;
    }
}

/* --- Strategy 2: quadruplets {a,b,c,d}. a*b and c*d are ANDed in round-1 chi
 * (pair rule), and the two products (ab)*(cd) are ANDed again in round-2 chi.
 * quad_partner: round-r chi partner of (L,z) (same rule as the pairs).
 * quad_phi:     where the product of (L,z)'s pair lands after one round
 *               (output lane (y-1, 2x+3y), z advanced by rho of the source lane).
 * quad_build:   from a free endpoint a, build b (partner of a), then choose c so
 *               that the product cd lands on the round-2 partner of ab, offset
 *               (+1,+1), which keeps ab and cd in different round-1 rows. */
static void quad_partner(int L, int z, int *Lp, int *zp) {
    int x = L % 5, y = L / 5;
    *Lp = (x + 1) % 5 + 5 * ((y + 1) % 5);
    *zp = (((z + rho_lane[L] - rho_lane[*Lp]) % 64) + 64) % 64;
}
static void quad_phi(int L, int z, int *Lr, int *zr) {
    int x = L % 5, y = L / 5;
    int xr = ((y - 1) % 5 + 5) % 5, yr = (2 * x + 3 * y) % 5;
    *Lr = xr + 5 * yr;
    *zr = (z + rho_lane[L]) % 64;
}
static void quad_build(int La, int za, int pos[4]) {
    int Lb, zb; quad_partner(La, za, &Lb, &zb);
    int Lpab, zpab; quad_phi(La, za, &Lpab, &zpab);
    int xq = (Lpab % 5 + 1) % 5, yq = (Lpab / 5 + 1) % 5;
    int Lq = xq + 5 * yq;
    int zq = (((zpab + rho_lane[Lpab] - rho_lane[Lq]) % 64) + 64) % 64;
    int yc = (xq + 1) % 5;
    int xc = (3 * yq + xq + 1) % 5;
    int Lc = xc + 5 * yc;
    int zc = (((zq - rho_lane[Lc]) % 64) + 64) % 64;
    int Ld, zd; quad_partner(Lc, zc, &Ld, &zd);
    pos[0] = 64 * La + za; pos[1] = 64 * Lb + zb;
    pos[2] = 64 * Lc + zc; pos[3] = 64 * Ld + zd;
}
/* Tile the cube with as many quadruplets as fit. d = 2^r-1 is odd, so the last
 * 3 variables are one pair plus one free variable. A quadruplet is rejected if
 * any of its 4 positions is outside the iv range or already used. */
static void gen_cube_quad(int *cube_idx, int d, int klen, int nlen) {
    unsigned char used[25 * 64]; memset(used, 0, sizeof used);
    int lo = klen, hi = klen + nlen, count = 0;
    while (count < d) {
        int remaining = d - count;
        if (remaining >= 4) {
            int placed = 0;
            for (int tries = 0; tries < 20000 && !placed; tries++) {
                int pa = lo + rand() % nlen, La = pa >> 6, za = pa & 63;
                int pos[4]; quad_build(La, za, pos);
                int ok = 1;
                for (int i = 0; i < 4 && ok; i++) {
                    if (pos[i] < lo || pos[i] >= hi || used[pos[i]]) ok = 0;
                    for (int j = 0; j < i && ok; j++) if (pos[i] == pos[j]) ok = 0;
                }
                if (!ok) continue;
                for (int i = 0; i < 4; i++) { used[pos[i]] = 1; cube_idx[count++] = pos[i] - klen; }
                placed = 1;
            }
            if (!placed) { int pa; do { pa = lo + rand() % nlen; } while (used[pa]); used[pa] = 1; cube_idx[count++] = pa - klen; }
        } else if (remaining >= 2) {
            int placed = 0;
            for (int tries = 0; tries < 20000 && !placed; tries++) {
                int pa = lo + rand() % nlen, La = pa >> 6, za = pa & 63;
                if (used[pa]) continue;
                int Lb, zb; quad_partner(La, za, &Lb, &zb);
                int pb = 64 * Lb + zb;
                if (pb < lo || pb >= hi || used[pb] || pb == pa) continue;
                used[pa] = used[pb] = 1; cube_idx[count++] = pa - klen; cube_idx[count++] = pb - klen;
                placed = 1;
            }
            if (!placed) { int pa; do { pa = lo + rand() % nlen; } while (used[pa]); used[pa] = 1; cube_idx[count++] = pa - klen; }
        } else {
            int pa; do { pa = lo + rand() % nlen; } while (used[pa]);
            used[pa] = 1; cube_idx[count++] = pa - klen;
        }
    }
}

/* Strategy 0: independent uniform-random distinct iv bits. */
static void gen_cube_random(int *cube_idx, int d, int nlen) {
    for (int i = 0; i < d; i++) {
        int cand, ok;
        do { cand = rand() % nlen; ok = 1; for (int q = 0; q < i; q++) if (cube_idx[q] == cand) ok = 0; } while (!ok);
        cube_idx[i] = cand;
    }
}
static void gen_cube(int *cube_idx, int d, int klen, int nlen, int src) {
    if      (src == 1) gen_cube_pairing(cube_idx, d, klen, nlen);
    else if (src == 2) gen_cube_quad(cube_idx, d, klen, nlen);
    else               gen_cube_random(cube_idx, d, nlen);
}

/* Self-test for cube_src=2: confirm each quadruplet's a*b*c*d product survives
 * 2 rounds (2-round, 4-bit cube-sum from an all-zero base; nonzero => present). */
static void quad_selftest(GpuCtx *g, int klen, int nlen, int nq) {
    static uint64_t qb[MAXTASK * 25];
    static int qpos[MAXTASK * KF1600_DMAX];
    static int qdarr[MAXTASK];
    static unsigned long long qtot[MAXTASK];
    static unsigned long long qst[MAXTASK * 25];
    if (nq > MAXTASK) nq = MAXTASK;
    int made = 0;
    while (made < nq) {
        int pa = klen + rand() % nlen, La = pa >> 6, za = pa & 63;
        int pos[4]; quad_build(La, za, pos);
        int ok = 1;
        for (int i = 0; i < 4 && ok; i++) {
            if (pos[i] < klen || pos[i] >= klen + nlen) ok = 0;
            for (int j = 0; j < i && ok; j++) if (pos[i] == pos[j]) ok = 0;
        }
        if (!ok) continue;
        for (int L = 0; L < 25; L++) qb[made * 25 + L] = 0;
        for (int i = 0; i < 4; i++) qpos[made * KF1600_DMAX + i] = pos[i];
        qdarr[made] = 4; qtot[made] = 16; made++;
    }
    int saved = g_invhalf; g_invhalf = 0;      /* this 2-round structural check uses RAW cube-sums */
    run_state_batch(g, qb, qpos, qdarr, qtot, nq, 2, qst);
    g_invhalf = saved;
    int pass = 0;
    for (int t = 0; t < nq; t++) {
        int nz = 0; for (int L = 0; L < 25; L++) if (qst[t * 25 + L]) { nz = 1; break; }
        if (nz) pass++;
    }
    printf("[quad self-test] %d/%d quadruplets: a*b*c*d present in the 2-round output ANF%s\n",
           pass, nq, pass == nq ? "  [OK]" : "  [WARNING: some failed]");
}

/* GF(2) solve of coefs*x = rhs over S vars. Returns rank; if rank==S the
 * unique solution is written to x. */
static int gf2_solve_bv(const BV *coefs, const uint8_t *rhs, int m, uint8_t *x) {
    static BV row[MAXTASK]; static uint8_t rr[MAXTASK];
    for (int i=0;i<m;i++){ row[i]=coefs[i]; rr[i]=rhs[i]&1; }
    int rank=0;
    for (int col=0; col<S && rank<m; col++) {
        int piv=-1;
        for (int i=rank;i<m;i++) if (bv_testbit(&row[i],col)) { piv=i; break; }
        if (piv<0) continue;
        BV tv=row[rank]; row[rank]=row[piv]; row[piv]=tv;
        uint8_t tr=rr[rank]; rr[rank]=rr[piv]; rr[piv]=tr;
        for (int i=0;i<m;i++) if (i!=rank && bv_testbit(&row[i],col)) { bv_xor(&row[i],&row[rank]); rr[i]^=rr[rank]; }
        rank++;
    }
    if (x) { memset(x,0,S);
        int r2=0;
        for (int col=0; col<S && r2<rank; col++)
            if (bv_testbit(&row[r2],col)) { x[col]=rr[r2]; r2++; }
    }
    return rank;
}

/* preprocessing state */
static uint8_t base_iv[1600];
static BV      coefs[MAXTASK];
static BV      basis[MAXTASK];
static int     pivotidx[MAXTASK];
static uint8_t cconst[MAXTASK];
static int     outbits[MAXTASK];
static int     cube_store[MAXTASK][64];

int main(int argc, char **argv) {
    int   nr      = (argc > 1) ? atoi(argv[1]) : 5;
    int   nlen        = (argc > 2) ? atoi(argv[2]) : (1344 - S);   /* IV bits = rate - s */
    unsigned seed     = (argc > 3) ? (unsigned)strtoul(argv[3], NULL, 10) : 20260722u;
    long  max_cubes   = (argc > 4) ? atol(argv[4]) : 200000;
    double max_sec    = (argc > 5) ? atof(argv[5]) : 1e12;
    int   qd          = (argc > 6) ? atoi(argv[6]) : 1;
    int   rank_target = (argc > 7) ? atoi(argv[7]) : S;
    long  STALL       = (argc > 8) ? atol(argv[8]) : 100000;
    int   cube_src    = (argc > 9) ? atoi(argv[9]) : 0;
    /* invhalf: 0 = normal (cube dim 2^{n_r}-1); 1 = degree reduction -- invert the
     * last round's iota,chi on the four complete rows and attack with cube dim
     * 2^{n_r-1}-1. A six-round invhalf run then matches a five-round normal one. */
    int   invhalf     = (argc > 10) ? atoi(argv[10]) : 0;
    const int klen = S, w = 64;
    const int RATE = 1344;                      /* b - c = 1600 - 256, the observable rate */
    g_invhalf = invhalf;
    if (invhalf && nr < 1) { fprintf(stderr, "invhalf needs n_r >= 1\n"); return 2; }
    const int d = invhalf ? (1 << (nr - 1)) - 1 : (1 << nr) - 1;
    /* observable output bits: the whole rate (lanes 0..20), or the four complete
     * rows (lanes 0..19 = 1280 bits) when the last half-round is inverted. */
    const int obits = invhalf ? 20 * 64 : RATE;

    if (nr < 1 || d > KF1600_DMAX) { fprintf(stderr, "n_r out of range (d must be <= %d)\n", KF1600_DMAX); return 2; }
    if (d < 1)                          { fprintf(stderr, "d < 1 (n_r too small for this mode)\n"); return 2; }
    if (nlen < d || klen + nlen > RATE) { fprintf(stderr, "nlen must be in [d, %d]: key+IV live in the rate, capacity stays 0\n", RATE - klen); return 2; }
    if (qd != 0 && qd != 1)             { fprintf(stderr, "qd must be 0 or 1\n"); return 2; }
    if (cube_src < 0 || cube_src > 2)   { fprintf(stderr, "cube_src must be 0, 1 or 2\n"); return 2; }
    if (rank_target > klen) rank_target = klen;

    setvbuf(stdout, NULL, _IONBF, 0);
    srand(seed);
    keccak_params P = { w, nr, klen, nlen, NULL, NULL };
    uint8_t secret[S];
    for (int j = 0; j < klen; j++) secret[j] = rand() & 1;
    BV skey; bv_zero(&skey);
    for (int j = 0; j < klen; j++) if (secret[j]) bv_setbit(&skey, j);
    memset(base_iv, 0, nlen);

    GpuCtx g; ctx_init(&g);
    { uint64_t bs[25] = {0}; int p0[KF1600_DMAX] = {0};
      int dd = 0; unsigned long long tt = 1; unsigned long long st[25];
      run_state_batch(&g, bs, p0, &dd, &tt, 1, nr, st); }   /* warm-up */

    /* ---- validation: GPU vs validated C reference on random points ---------- */
    int vN = 16, vfail = 0;
    int vd = d < 12 ? d : 12;
    double tv = now_sec();
    for (int t = 0; t < vN; t++) {
        uint8_t vkey[S];
        for (int j = 0; j < klen; j++) vkey[j] = rand() & 1;
        int ci[64];
        for (int i = 0; i < vd; i++) {
            int cand, ok;
            do { cand = rand() % nlen; ok = 1; for (int q = 0; q < i; q++) if (ci[q]==cand) ok = 0; } while (!ok);
            ci[i] = cand;
        }
        /* in invhalf mode the inverted value is defined only on lanes 0..19 */
        int ob = invhalf ? (rand() % obits) : (rand() % (25 * w));
        int gpu = gpu_cube_sum(&g, &P, vkey, base_iv, ci, vd, ob);
        int ref = invhalf ? ref_cube_sum_invhalf(&P, vkey, base_iv, ci, vd, ob)
                          : cube_sum(&P, vkey, base_iv, ci, vd, ob);
        if (gpu != ref) vfail++;
    }
    printf("[validate] GPU cube-sum vs C reference on %d random points (d=%d%s): %s  (%.2f s)\n",
           vN, vd, invhalf ? ", invhalf" : "", vfail ? "MISMATCH (bug)" : "all match", now_sec() - tv);

    if (cube_src == 2) quad_selftest(&g, klen, nlen, 100);

    printf("[config] stream cipher = first %d bits of Keccak-f[1600]^{n_r=%d}( key[%d] || IV[%d] || 0^256 ) : "
           "s=%d secret bits @0..%d | IV @%d..%d | capacity 256 bits fixed 0 | "
           "d=%d cube_src=%d qd=%d invhalf=%d obits=%d rank_target=%d seed=%u\n",
           obits, nr, klen, nlen, klen, klen - 1, klen, klen + nlen - 1,
           d, cube_src, qd, invhalf, obits, rank_target, seed);

    int cube_idx[64];

    /* -------------------------- PREPROCESSING (GPU, harvest) ---------------- */
    int m = 0; long cubes = 0, discarded = 0, harvested = 0;
    long since_gain = 0;
    /* Parameter bookkeeping (see the [params] line printed after the offline phase). */
    long useful = 0;            /* harvested cubes with >= 1 non-constant superpoly     */
    long nc_total = 0;          /* non-constant superpolys over all harvested cubes     */
    long sigma = 0;             /* non-constant superpolys collected up to the last rank gain */

    double t0 = now_sec();
    while (m < rank_target && cubes < max_cubes && now_sec() - t0 < max_sec) {
        cubes++;
        gen_cube(cube_idx, d, klen, nlen, cube_src);
        if (qd == 1 && allzero_constants(&g, &P, base_iv, cube_idx, d, nr, obits)) {
            discarded++;
            if (discarded % 50 == 0)
                printf("  ..%ld cubes tried, %ld discarded, %ld harvested, rank=%d/%d  (%.1fs)\n",
                       cubes, discarded, harvested, m, klen, now_sec() - t0);
            continue;
        }

        reconstruct_harvest(&g, &P, base_iv, cube_idx, d);
        harvested++;

        int gained = 0, nc_cube = 0;
        for (int b = 0; b < obits; b++) {                /* every observable output bit is a candidate */
            int L = b >> 6, z = b & 63;
            unsigned long long s0 = states[L];           /* task 0 (key=0), lane L */
            BV v; bv_zero(&v);
            for (int j = 0; j < klen; j++)
                if (((states[(size_t)(j + 1) * 25 + L] ^ s0) >> z) & 1ULL) bv_setbit(&v, j);
            if (bv_iszero(&v)) continue;
            nc_cube++; nc_total++;                       /* non-constant superpoly */
            if (m >= klen) continue;
            BV rr = v;
            for (int i = 0; i < m; i++) if (bv_testbit(&rr, pivotidx[i])) bv_xor(&rr, &basis[i]);
            if (bv_iszero(&rr)) continue;
            basis[m] = rr; pivotidx[m] = bv_lowbit(&rr);
            coefs[m] = v; cconst[m] = (uint8_t)((s0 >> z) & 1ULL); outbits[m] = b;
            memcpy(cube_store[m], cube_idx, sizeof(int) * (size_t)d);
            m++; gained++;
            sigma = nc_total;
        }
        if (nc_cube > 0) useful++;
        printf("cube %5ld (harvest %4ld): nc=%-5d rank+=%-4d rank=%d/%d  (%.1fs)\n",
               cubes, harvested, nc_cube, gained, m, klen, now_sec() - t0);
        if (gained) since_gain = 0;
        else if (++since_gain > STALL) { printf("[stall] %ld harvested cubes with no new rank; stopping.\n", STALL); break; }
    }
    double t_pre = now_sec() - t0;
    printf("[preprocess/GPU] n_r=%d d=%d s=%d nlen=%d qd=%d cube_src=%d : rank %d/%d from %ld cubes "
           "(%ld quick-discarded = %.1f%%, %ld harvested) in %.3f s\n",
           nr, d, klen, nlen, qd, cube_src, m, klen, cubes, discarded,
           cubes ? 100.0*(double)discarded/(double)cubes : 0.0, harvested, t_pre);
    /* Attack parameters of this run:
     *   mu       = useful / tried       lambda   = non-const superpolys per useful cube
     *   mu_h     = harvested / tried    lambda_h = non-const superpolys per harvested cube
     *   sigma    = non-const superpolys collected until the final rank was reached
     * Without quick discard (qd=0) every tried cube is harvested, so mu_h = 1. */
    printf("[params] tried=%ld discarded=%ld harvested=%ld useful=%ld | mu=%.4f mu_h=%.4f "
           "lambda=%.2f lambda_h=%.2f sigma=%ld (rank %d)\n",
           cubes, discarded, harvested, useful,
           cubes ? (double)useful / (double)cubes : 0.0,
           cubes ? (double)harvested / (double)cubes : 0.0,
           useful ? (double)nc_total / (double)useful : 0.0,
           harvested ? (double)nc_total / (double)harvested : 0.0,
           sigma, m);

    /* ------------------------------ ONLINE (GPU) ---------------------------- */
    double t1 = now_sec();
    static uint64_t bstates[MAXTASK * 25];
    static int posarr[MAXTASK * KF1600_DMAX];
    static int darr[MAXTASK], outarr[MAXTASK];
    static unsigned long long totarr[MAXTASK];
    static unsigned res[MAXTASK];
    for (int i = 0; i < m; i++) {
        build_base_state(&P, secret, base_iv, cube_store[i], d, &bstates[i * 25]);
        for (int k2 = 0; k2 < d; k2++) posarr[i * KF1600_DMAX + k2] = P.klen + cube_store[i][k2];
        darr[i] = d; outarr[i] = outbits[i]; totarr[i] = 1ULL << d;
    }
    static uint8_t rhs[MAXTASK]; int inv_ok = 1;
    if (m > 0) run_bit_batch(&g, bstates, posarr, darr, outarr, totarr, m, nr, res);
    for (int i = 0; i < m; i++) {
        rhs[i] = (uint8_t)((res[i] & 1) ^ cconst[i]);
        if (bv_parity_and(&coefs[i], &skey) != rhs[i]) inv_ok = 0;
    }
    double t_on = now_sec() - t1;
    printf("[online/GPU] %d equations in %.4f s; exactness invariant: %s\n",
           m, t_on, inv_ok ? "OK" : "VIOLATED (bug)");

    /* ------------------------------- SOLVE ---------------------------------- */
    double ts = now_sec();
    uint8_t x[S];
    int rank = gf2_solve_bv(coefs, rhs, m, x);
    double t_solve = now_sec() - ts;
    printf("[solve] rank=%d/%d in %.5f s (linear algebra)\n", rank, klen, t_solve);
    printf("[timing] offline=%.3f s | online=%.4f s | linear-algebra=%.5f s | total=%.3f s\n",
           t_pre, t_on, t_solve, t_pre + t_on + t_solve);

    if (rank == klen) {
        int correct = 1;
        for (int j = 0; j < klen; j++) if (x[j] != secret[j]) correct = 0;
        if (correct) printf("RESULT: FULL %d-bit KEY RECOVERED (GPU, f[1600]) -- matches secret [OK]\n", klen);
        else         printf("RESULT: full rank but MISMATCH -- bug!\n");
        return (correct && inv_ok && !vfail) ? 0 : 1;
    }
    int rem = klen - rank;
    printf("RESULT: partial -- %d/%d key bits constrained; remaining %d bits => 2^%d brute-force%s\n",
           rank, klen, rem, rem, rem <= 40 ? " (practical)" : " (INFEASIBLE)");
    return (inv_ok && !vfail) ? 0 : 1;
}
