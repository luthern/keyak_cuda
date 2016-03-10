#include <stdint.h>
#include <string.h>
#include "piston.h"
#include "defs.h"


void buffer_init(Buffer * b)
{
    memset(b, 0, sizeof(Buffer));
    // b->size = KEYAK_BUFFER_SIZE;
}

void piston_init(Piston * p, uint32_t Rs, uint32_t Ra)
{
    p->Rs = Rs;
    p->Ra = Ra;
    
    p->EOM = Ra;
    p->CryptEnd = Ra + 1;
    p->InjectStart = Ra + 2;
    p->InjectEnd = Ra + 3;
    
    memset(p->state, 0, KEYAK_STATE_SIZE);
}

