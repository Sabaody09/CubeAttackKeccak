# GPU-parallel cube attack on a round-reduced Keccak stream cipher (Keccak-f[1600]).
#
#   make                  build the attack for s = 128, 256 and 512 secret bits
#   make check            CPU self-tests (Keccak KATs, half-round inversion)
#   make CUDA_ARCH=sm_90  build for another GPU generation (default sm_80 = A100;
#                         sm_121 = GB10, sm_80 = A100, sm_90 = H100)
#
# The secret size s is a compile-time constant (one binary per size). The round
# count n_r, the cube-selection strategy, the quick-discard filter and the
# degree-reduction (invhalf) mode are all run-time arguments; see the README.

CC        ?= gcc
NVCC      ?= nvcc
CUDA_ARCH ?= sm_80
CFLAGS    := -O2 -Wall -Wextra -std=c11 -Isrc
NVFLAGS   := -O3 -arch=$(CUDA_ARCH) -Isrc -Igpu

OBJS := keccak.o keystream.o cube.o

.PHONY: all check clean
all: cube_attack_s128 cube_attack_s256 cube_attack_s512

keccak.o: src/keccak.c src/keccak.h
	$(CC) $(CFLAGS) -c $< -o $@
keystream.o: src/keystream.c src/keystream.h src/keccak.h
	$(CC) $(CFLAGS) -c $< -o $@
cube.o: src/cube.c src/cube.h src/keystream.h src/keccak.h
	$(CC) $(CFLAGS) -c $< -o $@

cube_attack_s128: gpu/cube_attack.cu gpu/kf1600.cuh $(OBJS)
	$(NVCC) $(NVFLAGS) -DS_BITS=128 $< $(OBJS) -o $@
cube_attack_s256: gpu/cube_attack.cu gpu/kf1600.cuh $(OBJS)
	$(NVCC) $(NVFLAGS) -DS_BITS=256 $< $(OBJS) -o $@
cube_attack_s512: gpu/cube_attack.cu gpu/kf1600.cuh $(OBJS)
	$(NVCC) $(NVFLAGS) -DS_BITS=512 $< $(OBJS) -o $@

# ---- CPU self-tests (no GPU needed) ----
keccak_kat: tests/keccak_kat.c src/keccak.c src/keccak.h
	$(CC) $(CFLAGS) -o $@ tests/keccak_kat.c src/keccak.c
halfround_test: tests/halfround_test.c src/keccak.c src/keystream.c src/keccak.h src/keystream.h
	$(CC) $(CFLAGS) -o $@ tests/halfround_test.c src/keccak.c src/keystream.c

check: keccak_kat halfround_test
	./keccak_kat
	./halfround_test

clean:
	rm -f $(OBJS) cube_attack_s128 cube_attack_s256 cube_attack_s512 keccak_kat halfround_test
