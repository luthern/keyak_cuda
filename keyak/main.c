#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "keyak.h"

int main(int argc, char * argv[])
{
    
    Keyak k;

    // lunar keyak
    keyak_init(&k,1600,12,256,128);

    printf("hello keyak\n");

    return 0;
}
