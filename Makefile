# Cube attack on round-reduced Keccak-MAC (Keccak-f[1600]), GPU-parallel.
#
#   make                 build both drivers (128- and 256-bit key)
#   make CUDA_ARCH=sm_90 build for another GPU generation (default: sm_80 = A100)

CC        ?= gcc
NVCC      ?= nvcc
CUDA_ARCH ?= sm_80
CFLAGS    := -O2 -Wall -Wextra -std=c11 -Isrc
NVFLAGS   := -O3 -arch=$(CUDA_ARCH) -Isrc -Igpu

OBJS := keccak.o mac.o cube.o

.PHONY: all clean
all: cube_attack_c128 cube_attack_c256

keccak.o: src/keccak.c src/keccak.h
	$(CC) $(CFLAGS) -c $< -o $@
mac.o: src/mac.c src/mac.h src/keccak.h
	$(CC) $(CFLAGS) -c $< -o $@
cube.o: src/cube.c src/cube.h src/mac.h src/keccak.h
	$(CC) $(CFLAGS) -c $< -o $@

cube_attack_c128: gpu/cube_attack.cu gpu/kf1600.cuh $(OBJS)
	$(NVCC) $(NVFLAGS) -DKLEN_BITS=128 $< $(OBJS) -o $@

cube_attack_c256: gpu/cube_attack.cu gpu/kf1600.cuh $(OBJS)
	$(NVCC) $(NVFLAGS) -DKLEN_BITS=256 $< $(OBJS) -o $@

clean:
	rm -f $(OBJS) cube_attack_c128 cube_attack_c256
