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

__global__ void piston_spark(uint8_t * state, uint8_t eom, uint8_t * offsets, uint8_t * dst, uint8_t amt)
{
    uint8_t piston = blockIdx.x;
    uint32_t stateoffset = piston * KEYAK_STATE_SIZE;
    
    if (eom && threadIdx.x == 0)
    {
        uint8_t offset = offsets == NULL ? 0 : offsets[piston];
        state[stateoffset + PISTON_EOM] ^= ( offset == 0 ) ? 0xff : offset;
    }

    PERMUTE((uint64_t*)(state + stateoffset));
    if (amt)
    {
        dst[piston * amt + threadIdx.x] = state[piston * KEYAK_STATE_SIZE + threadIdx.x];
    }
}

__global__ void piston_centralize_state(uint8_t * dst, uint8_t * state, uint8_t amt)
{
    uint8_t piston = blockIdx.x;
    dst[piston * amt + threadIdx.x] = state[piston * KEYAK_STATE_SIZE + threadIdx.x];
}

__device__ void _piston_inject(uint8_t * state, uint8_t * x, uint32_t size, uint8_t crypting, uint8_t doSpark)
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
    if (doSpark)
    {
        PERMUTE((uint64_t *)(state + statestart));
    }

}

// size is the size of entire meta data block for pistons to absorb
// size <= PISTON_RA * KEYAK_NUM_PISTONS
__global__ void piston_inject_seq(uint8_t * state, uint8_t * x, uint32_t size, uint8_t crypting, uint8_t doSpark)
{
    _piston_inject(state,x,size,crypting,doSpark);
}

// Copies from same buffer to all pistons states
// Allows diversify flag.
// size must not be bigger then RA
__global__ void piston_inject_uniform(uint8_t * state, uint8_t * x, uint32_t offset, uint8_t size, uint8_t dFlag, uint8_t sparkFlag)
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
    if (sparkFlag)
    {
        PERMUTE((uint64_t*)(state + statestart));
    }
}

__global__ void piston_crypt(   uint8_t * in, uint8_t * out, uint8_t * state,
                                uint32_t _amt, uint8_t unwrapFlag, uint8_t * _x, uint32_t _size, uint8_t crypting, uint8_t _doSpark,
                                uint32_t rs_total, uint32_t ra_total)
{
    int i = blockIdx.x * KEYAK_STATE_SIZE + threadIdx.x;
    int consuming = blockIdx.x * PISTON_RS + threadIdx.x;
    uint8_t piston = blockIdx.x;
    uint8_t do_spark;
    
    uint32_t rs_offset = 0, ra_offset = 0;
    
    int rs_leftover, ra_leftover, rs_amt, ra_amt;

    while(rs_offset < rs_total)
    {
        rs_leftover = (rs_total - rs_offset);
        ra_leftover = (ra_total - ra_offset);

        rs_amt = MIN(rs_leftover, PISTON_RS * KEYAK_NUM_PISTONS);
        ra_amt = MIN(ra_leftover, PISTON_RA * KEYAK_NUM_PISTONS);

        do_spark = ((rs_offset + rs_amt) < rs_total || (ra_offset + ra_amt) < ra_total);

        if (consuming < rs_amt)
        {
            out[consuming] = state[i] ^ in[consuming];
            state[i] = unwrapFlag ? in[consuming] : in[consuming] ^ state[i];

            // if its last byte for piston ...
            if ( threadIdx.x == PISTON_RS-1 || consuming == rs_amt - 1)
            {
                state[piston * KEYAK_STATE_SIZE + PISTON_CRYPT_END] ^= (threadIdx.x + 1);
            }
        }

        if (ra_amt)
        {
            /*if (blockIdx.x == 0 && threadIdx.x ==0) printf("PISTON_INJECT\n", rs_amt);*/
            _piston_inject(state, in + (PISTON_RS * KEYAK_NUM_PISTONS), ra_amt, crypting, do_spark);
        }
        else if (do_spark)
        {
            /*if (blockIdx.x == 0 && threadIdx.x ==0) printf("PISTON_PERMUTE\n", rs_amt);*/
            PERMUTE((uint64_t *)(state + blockIdx.x * KEYAK_STATE_SIZE));
        }


        rs_offset += rs_amt;
        ra_offset += ra_amt;
        
        out += rs_amt;
        in += KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS;
    }

}


/*__global__ piston_motorist( uint8_t blocks, uint8_t numBlocks, size_t sizeLeftover, size_t size_rs, size_t size_ra )*/
/*{*/
    /*uint8_t i = 0;*/

    /*for (i=0; i < numBlocks)*/
    /*{*/
        /*int i = blockIdx.x * KEYAK_STATE_SIZE + threadIdx.x;*/
        /*int consuming = blockIdx.x * PISTON_RS + threadIdx.x;*/
        /*uint8_t piston = blockIdx.x;*/
        /*if (consuming < amt)*/
        /*{*/
            /*out[consuming] = state[i] ^ in[consuming];*/
            /*state[i] = unwrapFlag ? in[consuming] : in[consuming] ^ state[i];*/

            /*// if its last byte for piston ...*/
            /*if ( threadIdx.x == PISTON_RS-1 || consuming == amt - 1)*/
            /*{*/
                /*state[piston * KEYAK_STATE_SIZE + PISTON_CRYPT_END] ^= (threadIdx.x + 1);*/
            /*}*/
        /*}*/

        /*if (size > 0)*/
        /*{*/
            /*_piston_inject(state, x, size, crypting, doSpark);*/
        /*}*/

    /*}*/
/*}*/







