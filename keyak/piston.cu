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
    memset(p->state, 0, KEYAK_STATE_SIZE);
}

__global__ void piston_spark(uint8_t * state, uint8_t eom, uint8_t  offset)
{
    uint8_t piston = blockIdx.x;
    uint32_t stateoffset = piston * KEYAK_STATE_SIZE;
    
    state[stateoffset + PISTON_EOM] = 0;
    if (eom)
    {
        state[stateoffset + PISTON_EOM] = ( offset == 0 ) ? 0xff : offset;
    }
    PERMUTE(state + stateoffset);

}

#if 0
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
#endif

void piston_get_tag(Piston * p, Buffer * T, uint32_t l)
{
    assert(l <= PISTON_RS);
    int i;
    for (i=0; i < l; i++)
    {
        buffer_put(T, p->state[i]);
    }
}
/*
#define GPU_ERROR(x)       D_HANDLE_ERROR(x,__LINE__)
__device__ void D_HANDLE_ERROR(cudaError_t e, int line)
{
    if (e != cudaSuccess)
    {
        printf("line: %d. gpu error ?\n", line);
    }
}
*/

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

// size is size of each data to copy/inject to piston state
// size <= Ra
// offset is the offset from each block in x to pull from
__global__ void piston_inject(uint8_t * state, uint8_t * x, uint32_t offset, uint8_t size, uint8_t crypting)
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
#if 0
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
#endif

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
        // printf("out[%d] ^= %d ^ %d\n",i,state[i],in[i]);
        // int piston = i / PISTON_RS;
        out[consuming] = state[i] ^ in[consuming];
        state[i] = unwrapFlag ? in[consuming] : in[consuming] ^ state[i];

        // if its last byte for piston ...
        if ( threadIdx.x == PISTON_RS-1 || consuming == amt - 1)
        {
            state[piston * KEYAK_STATE_SIZE + PISTON_CRYPT_END] ^= threadIdx.x;
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
