
#include <stdio.h>
#include <string.h>
#include "keyak.h"
#include "motorist.h"
#include "misc.h"


void keyak_init(Keyak* k, uint32_t b, uint32_t nr, uint32_t c, uint32_t t)
{

    k->W = MAX(b/25,8);
    k->c = c;

    motorist_init(&k->motorist, k->W, c, t);
    
    buffer_init(&k->T);
    buffer_init(&k->SUV);
}

void keyak_set_suv()
{

}


