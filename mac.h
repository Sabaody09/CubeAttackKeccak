#ifndef MAC_H
#define MAC_H

#include <stdint.h>
#include "keccak.h"

/*
 * Keccak-MAC over the round-reduced permutation. This is the "black box"
 * queried by the cube attack.
 *
 * Bit addressing uses the canonical FIPS-202 linear index:
 *   p = w*(5*y + x) + z,   lane L = 5*y + x = p / w,   z = p % w,
 *   with x = L % 5, y = L / 5, and bit z of lane A[x][y].
 */

/* Set / get a single state bit by linear index p in [0, 25*w). */
void state_set_bit(lane_t A[5][5], int w, int p, int bit);
int  state_get_bit(const lane_t A[5][5], int w, int p);

typedef struct {
    int w;           /* lane width in bits (8 for f[200], 64 for f[1600]) */
    int rounds;      /* reduced round count */
    int klen;        /* number of secret key bits */
    int nlen;        /* number of nonce (public) bits */
    const int *kpos; /* klen key-bit positions, or NULL => contiguous 0..klen-1 */
    const int *npos; /* nlen nonce-bit positions, or NULL => klen..klen+nlen-1  */
} mac_params;

/*
 * keccak_mac:
 *   1. zero state
 *   2. write P->klen key bits   at positions kpos (or default layout)
 *   3. write P->nlen nonce bits at positions npos (or default layout)
 *   4. apply P->rounds rounds of Keccak-f[P->w]
 *   5. read `olen` output bits at positions opos[0..olen-1] into out[] (each 0/1)
 *
 * key   : array of P->klen bits (values in {0,1})
 * nonce : array of P->nlen bits (values in {0,1})
 */
void keccak_mac(const mac_params *P, const uint8_t *key, const uint8_t *nonce,
                const int *opos, int olen, uint8_t *out);

#endif /* MAC_H */
