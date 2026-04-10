#!/bin/bash

for b in 1 2 10 20 10000 20000 30000 40000 50000; do
    echo "STUDYING" $b
    ./test_conipfv_kernel $b
    echo "DONE STUDYING" $b
done
