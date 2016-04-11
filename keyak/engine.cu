#include <string.h>
#include <assert.h>
#include "engine.h"
#include "misc.h"
#include "piston.h"

#include "utils.h"

// cuda does not support external linkage
#include "keccak.cu"
#include "piston.cu"

#if 1

void dump_state(Engine * e, int piston)
{
    uint8_t tmp[KEYAK_STATE_SIZE];

    HANDLE_ERROR(cudaMemcpy(tmp,e->p_state + piston * KEYAK_STATE_SIZE,
                            KEYAK_STATE_SIZE, cudaMemcpyDeviceToHost));
    dump_hex(tmp, sizeof(tmp));
}

void dump_hex_cuda(uint8_t * buf, uint32_t size)
{
    char tbuf[20000];
    assert( size <= sizeof(tbuf));
    HANDLE_ERROR(cudaMemcpy(tbuf, buf, size, cudaMemcpyDeviceToHost));
    int i;
    for(i = 0; i < size; i++)
    {
        printf("%02hhx", tbuf[i]);
    }
    printf("\n");

}

#endif

// move memory to gpu from cpu
uint8_t * to_gpu(Engine * e, uint8_t bufsel, uint8_t * buf1, size_t size1)
{

    assert( size1 <= sizeof(e->coal1));

    // double buffering
    uint8_t * gpubufs[2] = {e->coal1_gpu, e->coal2_gpu};
    uint8_t * gpubuf = gpubufs[ bufsel % 2];

    HANDLE_ERROR(cudaMemcpyAsync( gpubuf, buf1, size1, cudaMemcpyHostToDevice));
    return gpubuf;
}

// interleave 2 cpu buffers and make 1 copy to GPU
uint8_t * coalesce_gpu(Engine * e, Packet * pkt)
{
    int total_blocks;
    int i,j=0,l=0;

    memset(pkt->rs_sizes,0,sizeof(pkt->rs_sizes[0])*(KEYAK_GPU_BUF_SLOTS));
    memset(pkt->ra_sizes,0,sizeof(pkt->ra_sizes[0])*(KEYAK_GPU_BUF_SLOTS));

    if (pkt->input_offset < pkt->input_size)
    {
        for (i=0; i < KEYAK_GPU_BUF_SLOTS; i++)
        {
            uint32_t tocopy = MIN(PISTON_RS * KEYAK_NUM_PISTONS, pkt->input_size - pkt->input_offset);
            memmove(pkt->merged + j, pkt->input + pkt->input_offset, tocopy);
            pkt->rs_sizes[l++] = tocopy;
            j += (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS);
            pkt->input_offset += tocopy;
            if (pkt->input_offset == pkt->input_size)
            {
                break;
            }
        }
    }
    total_blocks = i+1;

    if (pkt->metadata_offset < pkt->metadata_size)
    {
        j=0,l=0;
        for (i=0; i < KEYAK_GPU_BUF_SLOTS; i++)
        {
            uint32_t tocopy = MIN(PISTON_RA * KEYAK_NUM_PISTONS, pkt->metadata_size - pkt->metadata_offset);
            memmove(pkt->merged + j + PISTON_RS * KEYAK_NUM_PISTONS, pkt->metadata + pkt->metadata_offset, tocopy);
            pkt->ra_sizes[l++] = tocopy;
            j += (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS);
            pkt->metadata_offset += tocopy;
            if (pkt->metadata_offset == pkt->metadata_size)
            {
                break;
            }
        }
    }
    if (i > total_blocks)
    {
        total_blocks = i+1;
    }
    // sanity check
    assert(total_blocks <= KEYAK_GPU_BUF_SLOTS);
    
    //printf("copying over %d blocks\n", total_blocks);
    HANDLE_ERROR(cudaMemcpyAsync( e->coal1_gpu, pkt->merged,
                total_blocks * KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS, cudaMemcpyHostToDevice));

    return e->coal1_gpu;
}

void engine_init(Engine * e, Piston * pistons)
{
    memset(e,0,sizeof(Engine));
    e->pistons = pistons;
    e->phase = EngineFresh;

    // TODO consider making these all one contiguous block or even different memories
    HANDLE_ERROR(cudaMalloc(&e->p_in, PISTON_RS * KEYAK_NUM_PISTONS ));

    HANDLE_ERROR(cudaMalloc(&e->p_out, PISTON_RS * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_state, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_tmp, KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_offsets, KEYAK_NUM_PISTONS ));
    
    HANDLE_ERROR(cudaMalloc(&e->coal1_gpu, sizeof(e->coal1)));
    HANDLE_ERROR(cudaMalloc(&e->coal2_gpu, sizeof(e->coal2)));

    HANDLE_ERROR(cudaMemset(e->p_state,0, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMemset(e->p_offsets,0,KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMemset(e->p_tmp,0,KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS ));
}

void engine_destroy(Engine * e)
{
    printf("engine_destroyed\n");
    cudaFree(e->p_in);
    cudaFree(e->p_out);
}

void engine_restart(Engine * e)
{
    e->phase = EngineFresh;

    HANDLE_ERROR(cudaMemset(e->p_in, 0, PISTON_RS * KEYAK_NUM_PISTONS ));

    HANDLE_ERROR(cudaMemset(e->p_out, 0, PISTON_RS * KEYAK_NUM_PISTONS ));

    HANDLE_ERROR(cudaMemset(e->p_state,0, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMemset(e->p_offsets,0,KEYAK_NUM_PISTONS ));
    //HANDLE_ERROR(cudaMemset(e->p_tmp,0,KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS ));
}

void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets)
{

    cudaMemcpyAsync(e->p_offsets, offsets, KEYAK_NUM_PISTONS, cudaMemcpyHostToDevice);


    piston_spark<<<KEYAK_NUM_PISTONS,1>>>
        (e->p_state, eom, e->p_offsets);

    memmove(e->Et, offsets, KEYAK_NUM_PISTONS);
}

void engine_get_tags(Engine * e, Buffer * T, uint8_t * L)
{
    assert(e->phase == EngineEndOfMessage);
    uint8_t i;
    engine_spark(e, 1, L);

    for (i = 0; i < KEYAK_NUM_PISTONS; i++)
    {
        if (L[i])
        {
            // TODO consider making one copy or making this async
            assert(L[i] <= PISTON_RS);
            HANDLE_ERROR(
                    cudaMemcpyAsync(T->buf + T->length,
                                e->p_state + i * KEYAK_STATE_SIZE,
                                L[i], cudaMemcpyDeviceToHost)
                    );

            T->length += L[i];
        }
    }
    e->phase = EngineFresh;
}

uint8_t offsets_zero[KEYAK_NUM_PISTONS];
void engine_precompute()
{
    memset(offsets_zero, 0, sizeof(offsets_zero));
}

void engine_inject(Engine * e, uint8_t * A, uint8_t isLeftovers,uint32_t amt)
{
    assert(
            e->phase == EngineCrypted ||
            e->phase == EngineEndOfCrypt ||
            e->phase == EngineFresh
            );
    uint8_t cryptingFlag = (
            e->phase == EngineCrypted ||
            e->phase == EngineEndOfCrypt
            );

    if (amt)
    {
        piston_inject_seq<<<KEYAK_NUM_PISTONS, PISTON_RA>>>
        (e->p_state, A, 0, amt, cryptingFlag);
    }

    if (e->phase == EngineCrypted || isLeftovers)
    {
        engine_spark(e,0, offsets_zero);
        e->phase = EngineFresh;
    }
    else
    {
        e->phase = EngineEndOfMessage;
    }
}

void engine_inject_collective(Engine * e, Buffer * X, uint8_t dFlag)
{
    assert(e->phase == EngineFresh);

    if (dFlag)
    {
        buffer_put(X,KEYAK_NUM_PISTONS);
        buffer_put(X,0);
    }

    // TODO should support variable length
    assert(X->length < KEYAK_BUFFER_SIZE);

    // copy collective to gpu
    HANDLE_ERROR(cudaMemcpyAsync(e->p_tmp,X->buf,
                X->length,
                cudaMemcpyHostToDevice));


    uint32_t i;
    for (i=0; i < X->length; i += PISTON_RA)
    {
        if ( i + PISTON_RA >= X->length)
        {
            piston_inject_uniform<<<KEYAK_NUM_PISTONS, PISTON_RA>>>(e->p_state,
                    e->p_tmp, i, X->length - i, dFlag);
        }
        else
        {
            piston_inject_uniform<<<KEYAK_NUM_PISTONS, PISTON_RA>>>(e->p_state,
                    e->p_tmp, i, PISTON_RA, 0);
            piston_spark<<<KEYAK_NUM_PISTONS,1>>>
                (e->p_state, 0, NULL);

        }

    }

    if (dFlag)
    {
        X->length -= 2;
    }


    e->phase = EngineEndOfMessage;
}

// I is a GPU owned buffer
// O is a CPU owned buffer
void engine_crypt(Engine * e, uint8_t * I, Buffer * O, uint8_t unwrapFlag, uint32_t amt)
{

    assert(e->phase == EngineFresh);

    // TODO is RISTON_RS i.e. 1-1 the best ratio here?

    piston_crypt<<<KEYAK_NUM_PISTONS,PISTON_RS>>>
        (I,e->p_out,e->p_state,amt, unwrapFlag);

    // Copy the output of pistons
    assert(O->length + amt < KEYAK_BUFFER_SIZE);
    HANDLE_ERROR(cudaMemcpyAsync(O->buf + O->length, e->p_out,
                amt,
                cudaMemcpyDeviceToHost));

    O->length += amt;

}


