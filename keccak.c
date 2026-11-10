#include "keccak.h"

/* Rho rotation offsets, indexed [x][y] (offsets for the standard w=64 case;
 * for smaller widths they are applied modulo w inside rotl()). */
static const int RHO[5][5] = {
    /* y=0  y=1  y=2  y=3  y=4 */
    {  0,  36,   3,  41,  18 }, /* x=0 */
    {  1,  44,  10,  45,   2 }, /* x=1 */
    { 62,   6,  43,  15,  61 }, /* x=2 */
    { 28,  55,  25,  21,  56 }, /* x=3 */
    { 27,  20,  39,   8,  14 }, /* x=4 */
};

/* Iota round constants (full 64-bit). For width w<64 the low w bits are used;
 * because the LFSR only sets bits at positions 2^j-1, truncation is exact. */
static const uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

static inline lane_t mask_w(lane_t x, int w) {
    return (w >= 64) ? x : (x & ((1ULL << w) - 1));
}

/* Left-rotate x by n bit positions within a w-bit lane. */
static inline lane_t rotl(lane_t x, int n, int w) {
    x = mask_w(x, w);
    n %= w;
    if (n == 0) return x;
    return mask_w((x << n) | (x >> (w - n)), w);
}

void keccak_permute(lane_t A[5][5], int w, int rounds) {
    const lane_t msk = (w >= 64) ? ~0ULL : ((1ULL << w) - 1);
    int x, y, r;

    /* Normalize input to w bits. */
    for (x = 0; x < 5; x++)
        for (y = 0; y < 5; y++)
            A[x][y] &= msk;

    for (r = 0; r < rounds; r++) {
        lane_t C[5], D[5], B[5][5];

        /* theta */
        for (x = 0; x < 5; x++)
            C[x] = A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4];
        for (x = 0; x < 5; x++)
            D[x] = C[(x + 4) % 5] ^ rotl(C[(x + 1) % 5], 1, w);
        for (x = 0; x < 5; x++)
            for (y = 0; y < 5; y++)
                A[x][y] ^= D[x];

        /* rho (rotate) + pi (reposition): B[y][(2x+3y)%5] = rot(A[x][y]) */
        for (x = 0; x < 5; x++)
            for (y = 0; y < 5; y++)
                B[y][(2 * x + 3 * y) % 5] = rotl(A[x][y], RHO[x][y], w);

        /* chi */
        for (y = 0; y < 5; y++)
            for (x = 0; x < 5; x++)
                A[x][y] = (B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])) & msk;

        /* iota */
        A[0][0] ^= (RC[r] & msk);
    }
}

int keccak_num_rounds(int w) {
    int l = 0;
    while (w > 1) { w >>= 1; l++; }
    return 12 + 2 * l;
}
