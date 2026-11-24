/*
 * Known-Answer Test for the reference Keccak-f[1600] permutation.
 *
 * We validate the permutation by building SHA3-256 on top of it (rate=1088,
 * capacity=512, domain suffix 0x06, pad10*1) and checking against the canonical
 * NIST test vectors. Passing SHA3-256("") and SHA3-256("abc") exercises every
 * step mapping (theta/rho/pi/chi/iota) and all 24 round constants.
 *
 * Usage:
 *   ./kat                       run built-in KAT self-test (exit 0 on success)
 *   ./kat --sha3-256 <hex>      print SHA3-256 of the hex-decoded input
 *                               (used to cross-check against Python hashlib)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "keccak.h"

static void bytes_to_lanes(const uint8_t s[200], lane_t A[5][5]) {
    for (int y = 0; y < 5; y++)
        for (int x = 0; x < 5; x++) {
            int idx = x + 5 * y;
            lane_t v = 0;
            for (int b = 0; b < 8; b++)
                v |= (lane_t)s[8 * idx + b] << (8 * b);
            A[x][y] = v;
        }
}

static void lanes_to_bytes(const lane_t A[5][5], uint8_t s[200]) {
    for (int y = 0; y < 5; y++)
        for (int x = 0; x < 5; x++) {
            int idx = x + 5 * y;
            lane_t v = A[x][y];
            for (int b = 0; b < 8; b++)
                s[8 * idx + b] = (uint8_t)(v >> (8 * b));
        }
}

static void keccak_f1600(uint8_t s[200]) {
    lane_t A[5][5];
    bytes_to_lanes(s, A);
    keccak_permute(A, 64, 24);
    lanes_to_bytes(A, s);
}

static void sha3_256(const uint8_t *msg, size_t len, uint8_t out[32]) {
    const size_t rate = 136; /* 1088 bits */
    uint8_t s[200];
    memset(s, 0, sizeof s);

    size_t i = 0;
    while (len - i >= rate) {
        for (size_t j = 0; j < rate; j++) s[j] ^= msg[i + j];
        keccak_f1600(s);
        i += rate;
    }
    /* final (partial) block with pad10*1 and SHA-3 domain suffix */
    uint8_t blk[136];
    memset(blk, 0, rate);
    size_t rem = len - i;
    for (size_t j = 0; j < rem; j++) blk[j] = msg[i + j];
    blk[rem]      ^= 0x06;
    blk[rate - 1] ^= 0x80;
    for (size_t j = 0; j < rate; j++) s[j] ^= blk[j];
    keccak_f1600(s);

    memcpy(out, s, 32);
}

static int digest_matches(const uint8_t d[32], const char *hex) {
    char buf[65];
    for (int k = 0; k < 32; k++) sprintf(buf + 2 * k, "%02x", d[k]);
    buf[64] = '\0';
    return strcmp(buf, hex) == 0;
}

int main(int argc, char **argv) {
    if (argc >= 3 && strcmp(argv[1], "--sha3-256") == 0) {
        const char *h = argv[2];
        size_t n = strlen(h) / 2;
        uint8_t *m = (uint8_t *)malloc(n ? n : 1);
        for (size_t k = 0; k < n; k++) {
            unsigned v = 0;
            sscanf(h + 2 * k, "%2x", &v);
            m[k] = (uint8_t)v;
        }
        uint8_t out[32];
        sha3_256(m, n, out);
        free(m);
        for (int k = 0; k < 32; k++) printf("%02x", out[k]);
        printf("\n");
        return 0;
    }

    int fail = 0;
    uint8_t out[32];

    sha3_256((const uint8_t *)"", 0, out);
    if (digest_matches(out, "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"))
        printf("PASS  SHA3-256(\"\")\n");
    else { printf("FAIL  SHA3-256(\"\")\n"); fail = 1; }

    sha3_256((const uint8_t *)"abc", 3, out);
    if (digest_matches(out, "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"))
        printf("PASS  SHA3-256(\"abc\")\n");
    else { printf("FAIL  SHA3-256(\"abc\")\n"); fail = 1; }

    if (fail) { printf("KAT FAILED\n"); return 1; }
    printf("ALL KAT PASSED\n");
    return 0;
}
