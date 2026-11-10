#include "cube.h"
#include <string.h>

int cube_sum(const mac_params *P, const uint8_t *key,
             const uint8_t *base_nonce,
             const int *cube_idx, int d, int out_bit_pos) {
    uint8_t nonce[1600];   /* full Keccak-f[1600] state upper bound */
    memcpy(nonce, base_nonce, (size_t)P->nlen);

    int acc = 0;
    uint64_t total = (uint64_t)1 << d;
    for (uint64_t m = 0; m < total; m++) {
        for (int i = 0; i < d; i++)
            nonce[cube_idx[i]] = (uint8_t)((m >> i) & 1);
        uint8_t bit;
        keccak_mac(P, key, nonce, &out_bit_pos, 1, &bit);
        acc ^= bit;
    }
    return acc & 1;
}

void superpoly_affine(const mac_params *P, const uint8_t *base_nonce,
                      const int *cube_idx, int d, int out_bit_pos,
                      uint8_t *coeff, uint8_t *c) {
    uint8_t key[1600];
    memset(key, 0, (size_t)P->klen);

    int c0 = cube_sum(P, key, base_nonce, cube_idx, d, out_bit_pos);
    *c = (uint8_t)c0;

    for (int j = 0; j < P->klen; j++) {
        key[j] = 1;
        int v = cube_sum(P, key, base_nonce, cube_idx, d, out_bit_pos);
        coeff[j] = (uint8_t)(v ^ c0);
        key[j] = 0;
    }
}

int superpoly_eval(const uint8_t *coeff, uint8_t c, const uint8_t *key, int klen) {
    int s = c & 1;
    for (int j = 0; j < klen; j++)
        s ^= (coeff[j] & key[j] & 1);
    return s & 1;
}
