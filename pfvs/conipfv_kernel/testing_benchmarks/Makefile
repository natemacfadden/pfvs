CFLAGS = -O3 -march=native -flto -mtune=native -funroll-loops
IFLAGS = -I.. -I$(CONDA_PREFIX)/include
LFLAGS = -L$(CONDA_PREFIX)/lib -lgmp -Wl,-rpath,$(CONDA_PREFIX)/lib
CC ?= clang

all: test_conipfv_kernel

test_conipfv_kernel: test_conipfv_kernel.c
	$(CC) $(CFLAGS) $(IFLAGS) $(LFLAGS) $< -o $@

clean:
	rm -f test_conipfv_kernel
