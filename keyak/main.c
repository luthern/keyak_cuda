#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "keyak.h"

int main(int argc, char * argv[])
{
    
    Keyak k;
    char suv[] = "avoneavckanesflie";
    char nonce[] = "k;owdemfew e lkj;lk";

    // lunar keyak
    keyak_init(&k,1600,12,256,128);

    keyak_set_suv(&k, suv, sizeof(suv));

    keyak_add_nonce(&k, nonce, sizeof(nonce));


    printf("hello keyak\n");

    return 0;
}
