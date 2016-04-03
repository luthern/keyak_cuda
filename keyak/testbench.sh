#!/bin/bash
i=10000
echo "Running for $i iterations"
python -c "print 'A'*4000" | ./keyak AAAAAAAAAAAAAAAAAAAAAAAAAAAAA out.data -m ABEF14230DFE -n abef912 -i $i
