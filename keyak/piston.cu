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
    b->length = len;
    b->buf = b->buf_stack;
    memmove(b->buf, data, len);
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
__device__ void piston_spark_dev(uint8_t * state, uint8_t eom, uint8_t * offsets, uint8_t * dst, uint8_t amt)
{
    uint8_t piston = blockIdx.x;
    uint32_t stateoffset = piston * KEYAK_STATE_SIZE;
    
    if (eom && threadIdx.x == 0)
    {
        uint8_t offset = offsets == NULL ? 0 : offsets[piston];
        state[stateoffset + PISTON_EOM] ^= ( offset == 0 ) ? 0xff : offset;
    }

    PERMUTE((uint64_t*)(state + stateoffset));

    int dsti = piston * amt + threadIdx.x;
    if (dsti < amt)
    {
        dst[dsti] = state[piston * KEYAK_STATE_SIZE + threadIdx.x];
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

__device__ void piston_inject_uniform_dev(uint8_t * state, uint8_t * x, uint32_t offset, uint8_t size, uint8_t dFlag, uint8_t sparkFlag)
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

__device__ void piston_crypt(   uint8_t * in, uint8_t * out, uint8_t * state,
                                uint32_t amt, uint8_t unwrapFlag, uint8_t * x, uint32_t size, uint8_t crypting, uint8_t doSpark)
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

    if (size)
    {
        _piston_inject(state, x, size, crypting, doSpark);
    }
    else if (doSpark)
    {
        PERMUTE((uint64_t *)(state + blockIdx.x * KEYAK_STATE_SIZE));
    }

}


__global__ void piston_crypt_super(   uint8_t * block, uint8_t * out, uint8_t * state_ext,
                                uint32_t rs_amt, uint8_t unwrapFlag, uint8_t * x, uint32_t ra_amt, uint8_t crypting, uint8_t doSpark,
                                uint32_t input_size, uint32_t input_bytes_processed, uint32_t metadata_size, uint32_t metadata_bytes_processed,
                                uint8_t * tag_out, uint8_t * tmp_ext)
{

    __shared__ uint8_t state[KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS];
    __shared__ uint8_t tmp[KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS];
    /*__shared__ uint8_t block[(1<<10)*40];*/

    /*int stride = (input_size + metadata_size) / (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS);*/

    int si = blockIdx.x * KEYAK_STATE_SIZE + threadIdx.x;
    state[si] = state_ext[si];

    if (threadIdx.x < PISTON_RS)
    {

        // CRYPT + INJECT //
        int out_offset = 0;

        int i;
        for (i=0; i < rs_amt / (PISTON_RS * KEYAK_NUM_PISTONS); i++)
        {
            int ra_left = MIN(ra_amt - metadata_bytes_processed, PISTON_RA * KEYAK_NUM_PISTONS);
            uint32_t offset = (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * i);

            /*timer_start(&tcrypt, "engine_crypt");*/

            uint8_t do_spark = ((input_bytes_processed + PISTON_RS * KEYAK_NUM_PISTONS) < input_size 
                    || (metadata_bytes_processed + ra_left) < metadata_size);

            piston_crypt(block + offset, out + out_offset, state, KEYAK_NUM_PISTONS * PISTON_RS, unwrapFlag,
                    block + offset + PISTON_RS * KEYAK_NUM_PISTONS, ra_left, crypting, do_spark);

            out_offset += KEYAK_NUM_PISTONS * PISTON_RS;

            input_bytes_processed += KEYAK_NUM_PISTONS * PISTON_RS;


            /*timer_accum(&tcrypt);*/


            metadata_bytes_processed += ra_left;

        }

        if ( rs_amt > input_bytes_processed)
        {
            int ra_left = MIN(ra_amt - metadata_bytes_processed, PISTON_RA * KEYAK_NUM_PISTONS);
            int rs_left = (rs_amt - input_bytes_processed);
            uint32_t offset = (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * i);

            /*timer_start(&tcrypt, "engine_crypt");*/

            uint8_t do_spark = ((input_bytes_processed + rs_left) < input_size 
                    || (metadata_bytes_processed + ra_left) < metadata_size);


            piston_crypt(block + offset, out + out_offset, state, rs_left, unwrapFlag,
                    block + offset + PISTON_RS * KEYAK_NUM_PISTONS, ra_left, crypting, do_spark);

            out_offset += rs_left;

            input_bytes_processed += rs_left;


            /*timer_accum(&tcrypt);*/

            metadata_bytes_processed += ra_left;
        }
        // END CRYPT + INJECT //

        // AUTHENTICATE //

        // make knot
        piston_spark_dev
            (state, 1, OFFSETS_CPRIME, tmp, KEYAK_CPRIME/8);
        // inject collective
        int col_size = KEYAK_NUM_PISTONS * KEYAK_CPRIME / 8;
        for (i=0; i < col_size; i += PISTON_RA)
        {
            if ( i + PISTON_RA >= col_size)
            {
                piston_inject_uniform_dev(state,
                        tmp, i, col_size - i, 0,0);
            }
            else
            {
                piston_inject_uniform_dev(state,
                        tmp, i, PISTON_RA, 0,1);
            }

        }

        // get tags
        piston_spark_dev
            (state, 1, OFFSETS_1TAG, tag_out, KEYAK_TAG_SIZE/8);




    }

}


