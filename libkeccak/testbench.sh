#!/bin/bash
i=10000
echo "Running for $i iterations"
python -c "print ('A'*1600)" > input.txt
./keyak AAAAAAAAAAAAAAAAAAAAAAAAAAAAA input.txt out.data -m ABEF14230DFE -n abef912 -i $i
rm out.data
