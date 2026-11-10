#ifndef KECCAK_H
#define KECCAK_H

#include <stdint.h>
#include <stddef.h>

/*
 * Round-reduced Keccak-f reference implementation.
 *
 * The permutation is parameterized by lane width `w` (runtime) so the SAME
 * binary can run both:
 *   - Keccak-f[200]  : w = 8   (25 lanes x 8 bits)   -- the toy target
 *   - Keccak-f[1600] : w = 64  (25 lanes x 64 bits)  -- real SHA-3 permutation
 *
 * State layout: A[x][y], x = column (0..4), y = row (0..4).
 * Each lane holds its value in the LOW `w` bits of a uint64_t; higher bits are 0.
 * The linear byte index of lane (x,y) is (x + 5*y), little-endian within a lane.
 *
 * Reduced rounds: round r (0-based) uses round constant RC[r]. Running `rounds`
 * rounds therefore applies RC[0..rounds-1]. With w=64, rounds=24 this is exactly
 * the standard SHA-3 permutation.
 */

typedef uint64_t lane_t;

/* In-place Keccak-f permutation.
 *   w      : lane width in bits (must be a power of two, 1..64; use 8 or 64).
 *   rounds : number of rounds to apply (0..24), using RC[0..rounds-1].
 */
void keccak_permute(lane_t A[5][5], int w, int rounds);

/* Standard (full) round count for a given lane width: 12 + 2*log2(w).
 * Returns 24 for w=64, 18 for w=8. */
int keccak_num_rounds(int w);

#endif /* KECCAK_H */
