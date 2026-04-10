Single-header style kernel (stb style).

To compile the testing code:
```
clang -O3 -march=native -flto -mtune=native -funroll-loops -I. testing_benchmarks/test_conipfv_kernel.c -o enum
```

Test with `./enum XXX` for some input dilation `XXX`, or via the bash scripts in `testing_benchmarks/`.
