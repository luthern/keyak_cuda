#include <stdint.h>
#include <string.h>
#include <assert.h>

#include <cuda.h>

#include "keccak.h"
#include "piston.h"
#include "defs.h"
#include "misc.h"


void buffer_init(Buffer * b, uint8_t * data, uint32_t len)
{
    memset(b, 0, sizeof(Buffer));
    memmove(b->buf, data, len);
    b->length = len;
}

__global__ void piston_spark(uint8_t * state, uint8_t eom, uint8_t * offsets)
{
    uint8_t piston = blockIdx.x;
    uint32_t stateoffset = piston * KEYAK_STATE_SIZE;
    
    if (eom && threadIdx.x == 0)
    {
        uint8_t offset = offsets == NULL ? 0 : offsets[piston];
        state[stateoffset + PISTON_EOM] ^= ( offset == 0 ) ? 0xff : offset;
    }
    PERMUTE((uint64_t*)(state + stateoffset));

}

__global__ void piston_centralize_state(uint8_t * dst, uint8_t * state, uint8_t amt)
{
    uint8_t piston = blockIdx.x;
    dst[piston * amt + threadIdx.x] = state[piston * KEYAK_STATE_SIZE + threadIdx.x];
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
        state[statestart + w + threadIdx.x]
            ^= x[i];
    }
    if (threadIdx.x == 0 && piston < KEYAK_NUM_PISTONS)
    {
        state[statestart + PISTON_INJECT_START] ^= w;

        uint16_t bitrate = cap * (piston + 1);
        if (bitrate <= size)
        {
            state[statestart + PISTON_INJECT_END] ^= cap;
        }
        else if ( size + cap > bitrate )
        {
            state[statestart + PISTON_INJECT_END] ^= w+(uint8_t)(size - cap * piston);
        }
        else
        {
            state[statestart + PISTON_INJECT_END] ^= w;
        }
    }
}

// Copies from same buffer to all pistons states
// Allows diversify flag.
// size must not be bigger then RA
__global__ void piston_inject_uniform(uint8_t * state, uint8_t * x, uint32_t offset, uint8_t size, uint8_t dFlag)
{
    uint8_t piston = blockIdx.x;
    uint32_t statestart = piston * KEYAK_STATE_SIZE;
    int i = threadIdx.x;

    if (i < offset + size)
    {
        if (threadIdx.x == 0)
        {
            state[statestart + PISTON_INJECT_START] 
                ^= 0;
            state[statestart + PISTON_INJECT_END] 
                ^= size;
        }
        if (dFlag)
        {
            if (i == size - 1)
            {
                state[statestart + 0 + threadIdx.x]
                    ^= piston;
            }
            else
            {
                state[statestart + 0 + threadIdx.x]
                    ^= x[offset + i];
            }
        }
        else
        {
            state[statestart + 0 + threadIdx.x]
                ^= x[offset + i];
        }
    }
}

__global__ void piston_crypt(   uint8_t * in, uint8_t * out, uint8_t * state,
                                uint32_t amt, uint8_t unwrapFlag)
{
    int i = blockIdx.x * KEYAK_STATE_SIZE + threadIdx.x;
    int consuming = blockIdx.x * PISTON_RS + threadIdx.x;
    uint8_t piston = blockIdx.x;
    if (consuming < amt)
    {
        out[consuming] = state[i] ^ in[consuming];
        state[i] = unwrapFlag ? in[consuming] : in[consuming] ^ state[i];

        // if its last byte for piston ...
        if ( threadIdx.x == PISTON_RS-1 || consuming == amt - 1)
        {
            state[piston * KEYAK_STATE_SIZE + PISTON_CRYPT_END] ^= (threadIdx.x + 1);
        }
    }
}


