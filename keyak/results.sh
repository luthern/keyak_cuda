#!/bin/bash


slots=1

python -c "print 'A'*4000000" > input.txt
#export CUDA_INC=.

for i in `seq 256` ; do

export BUF_SLOTS=$slots

make clean && make -j6

printf "slots: $slots,  " >&2

./keyak AAAAAAAAAAAAAAAAAAAAAAAAAAAAA input.txt out.data -m ABEF14230DFE -n abef912 -i $i -s $slots

slots=$(($slots+1))

done
