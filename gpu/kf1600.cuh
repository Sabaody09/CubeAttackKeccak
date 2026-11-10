#ifndef KF1600_CUH
#define KF1600_CUH

/*
 * Shared per-point evaluation for round-reduced Keccak-f[1600] (64-bit lanes),
 * used identically on host and device. Mirrors ref/keccak.c with w=64 (full rho
 * offsets and 64-bit round constants; no masking needed). Header-only /
 * `static inline` so each TU gets its own copy; the __global__ kernels live in
 * the .cu file.
 *
 * Bit indexing: p = 64*(5y+x) + z, so lane L = p>>6 = 5y+x, z = p&63,
 * x = L%5, y = L/5. base_state[L] holds the full 64-bit lane L.
 */

#include <cstdint>

#define KF1600_DMAX 64   /* max supported cube dimension */

__host__ __device__ static inline uint64_t rotl64(uint64_t x, int n) {
    n &= 63;
    if (n == 0) return x;
    return (x << n) | (x >> (64 - n));
}

__host__ __device__ static inline void permute64(uint64_t A[5][5], int rounds) {
    const uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
        0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
        0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL };
    const int RHO[5][5] = {
        { 0,36, 3,41,18},{ 1,44,10,45, 2},{62, 6,43,15,61},
        {28,55,25,21,56},{27,20,39, 8,14} };
    for (int r = 0; r < rounds; r++) {
        uint64_t C[5], D[5], B[5][5];
        for (int x = 0; x < 5; x++)
            C[x] = A[x][0]^A[x][1]^A[x][2]^A[x][3]^A[x][4];
        for (int x = 0; x < 5; x++)
            D[x] = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++) A[x][y] ^= D[x];
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                B[y][(2*x+3*y)%5] = rotl64(A[x][y], RHO[x][y]);
        for (int y = 0; y < 5; y++)
            for (int x = 0; x < 5; x++)
                A[x][y] = B[x][y] ^ ((~B[(x+1)%5][y]) & B[(x+2)%5][y]);
        A[0][0] ^= RC[r];
    }
}

__host__ __device__ static inline int eval_point64(const uint64_t *base_state,
        const int *pos, int d, unsigned long long m, int rounds, int out_bit_pos) {
    uint64_t A[5][5];
    for (int L = 0; L < 25; L++) A[L % 5][L / 5] = base_state[L];
    for (int i = 0; i < d; i++)
        if ((m >> i) & 1ULL) { int p = pos[i]; A[(p>>6)%5][(p>>6)/5] |= (1ULL << (p & 63)); }
    permute64(A, rounds);
    int L = out_bit_pos >> 6, z = out_bit_pos & 63;
    return (int)((A[L % 5][L / 5] >> z) & 1ULL);
}

#endif /* KF1600_CUH */
