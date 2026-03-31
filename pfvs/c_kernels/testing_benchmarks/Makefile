CFLAGS = -O3 -march=native -flto -mtune=native -funroll-loops
IFLAGS = -I.. -I$(CONDA_PREFIX)/include
LFLAGS = -L$(CONDA_PREFIX)/lib -lgmp -Wl,-rpath,$(CONDA_PREFIX)/lib

all: test_coni_kernel test_pvec_kernel

test_coni_kernel: test_coni_kernel.c
	clang $(CFLAGS) $(IFLAGS) $(LFLAGS) $< -o $@

test_pvec_kernel: test_pvec_kernel.c
	clang $(CFLAGS) $(IFLAGS) $< -o $@

clean:
	rm -f test_coni_kernel test_pvec_kernel
