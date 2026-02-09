#!/bin/bash

#for b in 100 200 300 400; do
for b in 100 200 300 400; do
	echo "STUDYING" $b
    ./enum $b
    echo "DONE STUDYING" $b
done
