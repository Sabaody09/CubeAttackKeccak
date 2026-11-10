#include "mac.h"

void state_set_bit(lane_t A[5][5], int w, int p, int bit) {
    int L = p / w, z = p % w;
    int x = L % 5, y = L / 5;
    lane_t m = (lane_t)1 << z;
    if (bit & 1) A[x][y] |=  m;
    else         A[x][y] &= ~m;
}

int state_get_bit(const lane_t A[5][5], int w, int p) {
    int L = p / w, z = p % w;
    int x = L % 5, y = L / 5;
    return (int)((A[x][y] >> z) & 1);
}

void keccak_mac(const mac_params *P, const uint8_t *key, const uint8_t *nonce,
                const int *opos, int olen, uint8_t *out) {
    lane_t A[5][5];
    for (int x = 0; x < 5; x++)
        for (int y = 0; y < 5; y++)
            A[x][y] = 0;

    for (int i = 0; i < P->klen; i++) {
        int p = P->kpos ? P->kpos[i] : i;
        state_set_bit(A, P->w, p, key[i]);
    }
    for (int i = 0; i < P->nlen; i++) {
        int p = P->npos ? P->npos[i] : (P->klen + i);
        state_set_bit(A, P->w, p, nonce[i]);
    }

    keccak_permute(A, P->w, P->rounds);

    for (int i = 0; i < olen; i++)
        out[i] = (uint8_t)state_get_bit(A, P->w, opos[i]);
}
