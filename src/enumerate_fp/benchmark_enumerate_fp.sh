#!/bin/bash

for b in 1 10 20 30 40 50; do
	echo "STUDYING" $b
    ./enum $b
    echo "DONE STUDYING" $b
done
