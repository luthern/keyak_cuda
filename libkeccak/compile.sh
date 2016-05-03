#!/bin/bash
gcc -O3 -o keyak main.c misc.c Keyakv2.c Motorist.c KeccakP-1600-times8-on1.c KeccakP-1600-opt64.c KeccakP-1600-times4-on1.c KeccakP-1600-times2-on1.c KeccakP-800-opt32.c -lcrypto
