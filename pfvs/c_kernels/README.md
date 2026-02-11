Kernels are using the stb style... single file.


For the `pvec_kernel`, the following command compiles the testing code
`
clang -O3 -march=native -flto -mtune=native -funroll-loops -I. testing_benchmarks/test_pvec_kernel.c -o enum
`

For the `coni_kernel`, the following command compiles the testing code
`
clang -O3 -march=native -flto -mtune=native -funroll-loops -I. testing_benchmarks/test_coni_kernel.c -o enum
`

These codes can then be tested either with `./enum XXX` for some input dilation `XXX` or via the bash scripts.
