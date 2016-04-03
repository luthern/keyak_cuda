#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <cuda.h>

#include "piston.h"
#include "defs.h"
#include "misc.h"


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
    // memset(p->state, 0, KEYAK_STATE_SIZE);
}

__global__ void piston_spark(uint8_t * state, uint8_t eom, uint8_t * offsets)
{
    uint8_t piston = blockIdx.x;
    uint32_t stateoffset = piston * KEYAK_STATE_SIZE;
    
    if (eom)
    {
        uint8_t offset = offsets == NULL ? 0 : offsets[piston];
        //printf("piston %d offset %d\n",piston,offset);
        state[stateoffset + PISTON_EOM] ^= ( offset == 0 ) ? 0xff : offset;
    }
    PERMUTE(state + stateoffset);

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

// make consecutive copies of memory
// for each piston
__global__ void dup_for_pistons(uint8_t * mem, size_t size, uint8_t dFlag)
{
    if (blockIdx.x == 0 && threadIdx.x > 0 && threadIdx.x < KEYAK_NUM_PISTONS)
    {
        memcpy(mem + (KEYAK_BUFFER_SIZE * threadIdx.x),
                                mem, size);
        if (dFlag)
        {
            *(mem + (KEYAK_BUFFER_SIZE * threadIdx.x) + size - 1) = threadIdx.x;
        }
    }

}

// size is the size of entire meta data block for pistons to absorb
// size <= PISTON_RA * KEYAK_NUM_PISTONS
// TODO consider crypting flag
__global__ void piston_inject_seq(uint8_t * state, uint8_t * x, uint32_t offset, uint32_t size, uint8_t crypting)
{
    uint8_t piston = blockIdx.x;
        
    uint8_t w = crypting ? PISTON_RS : 0;
    uint8_t cap = (PISTON_RA - w);
    int i = cap * piston + threadIdx.x;
    uint32_t statestart = piston * KEYAK_STATE_SIZE;

    if ( i < size)
    {
        //printf("byte %d injected\n", i);
        state[statestart + w + threadIdx.x]
            ^= x[i];
    }
    if (threadIdx.x == 0 && piston < KEYAK_NUM_PISTONS)
    {
        state[statestart + PISTON_INJECT_START] ^= w;

        uint16_t bitrate = cap * (piston + 1);
        if (bitrate <= size)
        {
            // printf("piston %d injected %d bytes\n", piston, w+cap);
            state[statestart + PISTON_INJECT_END] ^= cap;
        }
        else if ( size + cap > bitrate )
        {
            state[statestart + PISTON_INJECT_END] ^= w+(uint8_t)(size - cap * piston);
            //printf("piston %d ended with %d bytes\n", piston, size - cap * piston);
        }
        else
        {
            state[statestart + PISTON_INJECT_END] ^= w;
        }
    }
}
// size is size of each data to copy/inject to piston state
// size <= PISTON_RA
// offset is the offset from each block in x to pull from
// TODO consider crypting flag
__global__ void piston_inject_uniform(uint8_t * state, uint8_t * x, uint32_t offset, uint8_t size, uint8_t crypting)
{
    uint8_t piston = blockIdx.x;
    uint32_t statestart = piston * KEYAK_STATE_SIZE;
    int i = piston * KEYAK_BUFFER_SIZE + threadIdx.x;

    if (i < KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS)
    {
        uint8_t w = crypting ? PISTON_RS : 0;
        if (threadIdx.x == 0)
        {
            state[statestart + PISTON_INJECT_START] 
                ^= w;
            state[statestart + PISTON_INJECT_END] 
                ^= size;
        }
        state[statestart + w + threadIdx.x]
            ^= x[offset + i];
    }

}

#define CRYPT_SIZE                      (PISTON_RS * KEYAK_NUM_PISTONS)
#define MAX_CUDA_THREADS_PER_BLOCK      1024

__global__ void piston_crypt(   uint8_t * in, uint8_t * out, uint8_t * state,
                                uint32_t amt, uint8_t unwrapFlag)
{
    int i = blockIdx.x * KEYAK_STATE_SIZE + threadIdx.x;
    int consuming = blockIdx.x * PISTON_RS + threadIdx.x;
    uint8_t piston = blockIdx.x;
    if (consuming < amt)
    {
        //if (threadIdx.x==0) printf("piston %d start %d  \n", piston, consuming);
        // printf("out[%d] ^= %d ^ %d\n",i,state[i],in[i]);
        // int piston = i / PISTON_RS;
        out[consuming] = state[i] ^ in[consuming];
        state[i] = unwrapFlag ? in[consuming] : in[consuming] ^ state[i];

        // if its last byte for piston ...
        if ( threadIdx.x == PISTON_RS-1 || consuming == amt - 1)
        {
            state[piston * KEYAK_STATE_SIZE + PISTON_CRYPT_END] ^= (threadIdx.x + 1);
        }
    }
}


