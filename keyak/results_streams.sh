#!/bin/bash


slots=74
export CUDA_INC=.
export BUF_SLOTS=$slots
bytes=0
make clean && make -j6

for i in `seq 256` ; do

python -c "print 'A'*2500000" > input.txt
./keyak AAAAAAAAAAAAAAAAAAAAAAAAAAAAA input.txt out.data -m ABEF14230DFE -n abef912 -i 20 -s $i

done
