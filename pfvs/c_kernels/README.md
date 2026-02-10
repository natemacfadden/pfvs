Compile either like this
`
clang -O3 -march=native -flto -mtune=native -funroll-loops \
    -I coni_kernel/include \
    coni_kernel/main.c \
    coni_kernel/src/*.c \
    -o enum
`
