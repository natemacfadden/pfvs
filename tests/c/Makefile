PREFIX ?= $(or $(CONDA_PREFIX),/usr)
CC     ?= cc
CFLAGS ?= -O3 -march=native -g
IFLAGS  = -I../../pfvs/fp_kernel -I$(PREFIX)/include
LFLAGS  = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib -lgmp -lm

all: fp_kernel_cli fp_kernel_cli_stats

fp_kernel_cli: fp_kernel_cli.c ../../pfvs/fp_kernel/fp_kernel.h
	$(CC) -std=c11 $(CFLAGS) -Wall -Wextra $(IFLAGS) $< -o $@ $(LFLAGS)

fp_kernel_cli_stats: fp_kernel_cli.c ../../pfvs/fp_kernel/fp_kernel.h
	$(CC) -std=c11 $(CFLAGS) -DFPK_STATS -Wall -Wextra $(IFLAGS) $< -o $@ $(LFLAGS)

clean:
	rm -f fp_kernel_cli fp_kernel_cli_stats

.PHONY: all clean
