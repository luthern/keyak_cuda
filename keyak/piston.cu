#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <cuda.h>

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
}

void piston_init(Piston * p)
{
    memset(p->state, 0, KEYAK_STATE_SIZE);
}

void piston_restart(Piston * p)
{
    memset(p->state, 0, KEYAK_STATE_SIZE);
}

void piston_spark(Piston * p, uint8_t eom, uint8_t offset)
{
    if (eom)
    {
        p->state[PISTON_EOM] ^= ( offset == 0 ) ? 0xff : offset;
    }
    else
    {
        p->state[PISTON_EOM] ^= 0;
    }

    PERMUTE(p->state);
    // TODO add permutation call here
    // py-ref: self.state = self.f.apply(self.state)
}

void piston_get_tag(Piston * p, Buffer * T, uint32_t l)
{
    assert(l <= PISTON_RS);
    int i;
    for (i=0; i < l; i++)
    {
        buffer_put(T, p->state[i]);
    }
}

void piston_inject(Piston * p, Buffer * x, uint8_t crypting)
{
    uint8_t w = crypting ? PISTON_RS : 0;
    p->state[PISTON_INJECT_START] ^= w;

    while(buffer_has_more(x) && w < PISTON_RA)
    {
        p->state[w++] ^= buffer_get(x); 
    }
    p->state[PISTON_INJECT_END] ^= w;
}

#define CRYPT_SIZE                      (PISTON_RS * KEYAK_NUM_PISTONS)
#define MAX_CUDA_THREADS_PER_BLOCK      1024

__global__ void piston_crypt(uint8_t * in, uint8_t * out, uint8_t * state, uint32_t amt, uint8_t unwrapFlag)
{
    int i = blockIdx.x * MAX_CUDA_THREADS_PER_BLOCK + threadIdx.x;
    if (i < amt)
    {
        // printf("out[%d] ^= %d ^ %d\n",i,state[i],in[i]);
        // int piston = i / PISTON_RS;
        out[i] = state[i] ^ in[i];
        state[i] = unwrapFlag ? in[i] : in[i] ^ state[i];

        // if its last byte for piston ...
        if ( i % PISTON_RS == PISTON_RS-1 || i == amt - 1)
        {
            state[PISTON_CRYPT_END] ^= i % PISTON_RS;
        }
    } 
}
#if 0
void piston_crypt(Piston * p, Buffer * I, Buffer * O, uint8_t w,
        uint8_t unwrapFlag)
{
    while(buffer_has_more(I) && w < PISTON_RS)
    {
        uint8_t x = buffer_get(I);
        buffer_put(O, p->state[w] ^ x);
        p->state[w] = unwrapFlag ? x : p->state[w] ^ x;
        w++;
    }
    p->state[PISTON_CRYPT_END] ^= w;
}
#endif
