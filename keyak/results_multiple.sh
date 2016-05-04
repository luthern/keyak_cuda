#!/bin/bash


export CUDA_INC=.

for j in `seq 1 8 96` ; do

    export BUF_SLOTS=$j
    make clean && make -j6


    for i in `seq 10 5 100` ; do

        python -c "print 'A'*2500000" > input.txt
        ./keyak AAAAAAAAAAAAAAAAAAAAAAAAAAAAA input.txt out.data -m ABEF14230DFE -n abef912 -i 10 -s $i

    done

    echo
    echo

done
