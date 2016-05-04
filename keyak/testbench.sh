#!/bin/bash
i=$((5))
echo "Running for $i iterations"
python -c "print 'A'*3000000" > input.txt
./keyak AAAAAAAAAAAAAAAAAAAAAAAAAAAAA input.txt out.data -m ABEF14230DFE -n abef912 -i $i -s 50
