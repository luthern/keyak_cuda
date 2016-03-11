#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "piston.h"
#include "defs.h"
#include "misc.h"
#include "keccak.h"


void buffer_init(Buffer * b, uint8_t * data, uint32_t len)
{
    memset(b, 0, sizeof(Buffer));
    if (data != NULL)
    {
        while(len--)
        {
            buffer_put(b, *data++);
        }
    }
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

void piston_spark(Piston * p, uint8_t eom, uint8_t offset)
{
    if (eom)
    {
        p->state[p->EOM] ^= ( offset == 0 ) ? 0xff : offset;
    }
    else
    {
        p->state[p->EOM] ^= 0;
    }

    PERMUTE(p->state);
    // TODO add permutation call here
    // py-ref: self.state = self.f.apply(self.state)
}

void piston_get_tag(Piston * p, Buffer * T, uint32_t l)
{
    assert(l <= p->Rs);
    int i;
    for (i=0; i < l; i++)
    {
        buffer_put(T, p->state[i]);
    }
}

void piston_inject(Piston * p, Buffer * x, uint8_t crypting)
{
    uint8_t w = crypting ? p->Rs : 0;
    p->state[p->InjectStart] ^= w;

    while(buffer_has_more(x) && w < p->Ra)
    {
        p->state[w++] ^= buffer_get(x); 
    }
    p->state[p->InjectEnd] ^= w;
}

void piston_crypt(Piston * p, Buffer * I, Buffer * O, uint8_t w,
        uint8_t unwrapFlag)
{
    while(buffer_has_more(I) && w < p->Rs)
    {
        uint8_t x = buffer_get(I);
        buffer_put(O, p->state[w] ^ x);
        p->state[w] = unwrapFlag ? x : p->state[w] ^ x;
        w++;
    }
    p->state[p->CryptEnd] ^= w;
}
