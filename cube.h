#ifndef CUBE_H
#define CUBE_H

#include <stdint.h>
#include "mac.h"

/*
 * Cube machinery for the Keccak-MAC cube attack.
 *
 * KEY FACT (why there is no linearity test): after `rounds` rounds the output
 * bit has algebraic degree <= 2^rounds. A cube of dimension d = 2^rounds - 1
 * therefore has a superpoly of degree <= 2^rounds - d = 1, i.e. it is AFFINE in
 * the remaining variables. So we reconstruct it directly instead of testing it.
 *
 * Cube variables are given as indices into the nonce array (0..nlen-1). All
 * non-cube nonce bits are held at `base_nonce`; the key is held fixed per call.
 */

/* Parity (XOR) of a single MAC output bit over the whole 2^d cube. */
int cube_sum(const mac_params *P, const uint8_t *key,
             const uint8_t *base_nonce,
             const int *cube_idx, int d, int out_bit_pos);

/* Reconstruct the affine superpoly over the key bits:
 *   p(key) = c XOR ( XOR_j coeff[j] & key[j] ).
 * coeff must have room for P->klen entries. Costs (klen+1) cube sums. */
void superpoly_affine(const mac_params *P, const uint8_t *base_nonce,
                      const int *cube_idx, int d, int out_bit_pos,
                      uint8_t *coeff, uint8_t *c);

/* Evaluate a reconstructed affine superpoly at a given key. */
int superpoly_eval(const uint8_t *coeff, uint8_t c, const uint8_t *key, int klen);

#endif /* CUBE_H */
