#!/bin/bash

for b in 2 3 4 5 6 7; do
	echo "STUDYING" $b
    ./enum $b $b $b $b $b $b $b
    echo "DONE STUDYING" $b
done
