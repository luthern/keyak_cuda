#!/bin/bash


slots=1

for i in `seq 256` ; do

export BUF_SLOTS=$slots
export CUDA_INC=.
make clean && make -j6
printf "slots: $slots,  " >&2
./testbench.sh
slots=$(($slots+1))

done
