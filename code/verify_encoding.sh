#!/bin/bash

#OUT_NAME=reproduction-test
OUT_NAME=$1

fpath=/scratch/gpfs/eham/247-encoding-updated/minimal_reproduction/results/$OUT_NAME-hs
echo $fpath

for i in $(seq 1 1 48); do
	COUNT="$(ls $fpath$i/777/ | wc -l)"
	if [[ $COUNT != 161 ]]
	then
	echo "$i"
		echo $COUNT
	fi
done
