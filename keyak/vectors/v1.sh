#!/bin/bash
#
# self test for Lunar Keyak functionality
# from:
#   https://github.com/gvanas/KeccakCodePackage/blob/8e8cb0da8941e21d90fcd524036e21efea4151ad/CAESAR/LunarKeyak/selftest.c
#

key=5a4b3c2d1e0f00f1e2d3c4b5a6978879
nonce=6b4c2d0eefd0b19272533415f6d7b8990000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
metadata=32f3b47535f6
pt=e465e566e667e7
ciphertext=20fec6154502c4776b6a02bad7f9d331c96b626c49daf2

printf $pt | xxd -r -p | ../keyak $key ciphertext -n $nonce -m $metadata > /dev/null

calc_ciphertext=$(cat ciphertext | xxd -p ciphertext)

if [[ "$ciphertext" = "$calc_ciphertext" ]] ;
then

    echo "1 iteration Success"

else

    echo "FAIL"
    echo " $ciphertext != $calc_ciphertext "

fi


printf $pt | xxd -r -p | ../keyak $key ciphertext -n $nonce -m $metadata -i 100 > /dev/null

calc_ciphertext=$(cat ciphertext | xxd -p ciphertext)

if [[ "$ciphertext" = "$calc_ciphertext" ]] ;
then

    echo "100 iteration Success"

else

    echo "FAIL"
    echo " $ciphertext != $calc_ciphertext "

fi
