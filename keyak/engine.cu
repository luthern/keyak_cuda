#include <string.h>
#include <assert.h>
#include "engine.h"
#include "misc.h"
#include "piston.h"

#include "utils.h"

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


__device__ __constant__ uint8_t OFFSETS_CPRIME[8] = {KEYAK_CPRIME/8, KEYAK_CPRIME/8, KEYAK_CPRIME/8, KEYAK_CPRIME/8,
                                                       KEYAK_CPRIME/8, KEYAK_CPRIME/8, KEYAK_CPRIME/8, KEYAK_CPRIME/8};

__device__ __constant__ uint8_t OFFSETS_1TAG[8] = {KEYAK_TAG_SIZE/8, 0, 0, 0, 0, 0, 0, 0};
__device__ __constant__ uint8_t OFFSETS_ZERO[8] = {0, 0, 0, 0, 0, 0, 0, 0};

// cuda does not support external linkage
#include "keccak.cu"
#include "piston.cu"



// interleave 2 cpu buffers and make 1 copy to GPU
uint8_t * coalesce_gpu(Engine * e, Packet * pkt)
{
    int total_blocks;
    int i,j=0,l=0;

    memset(pkt->rs_sizes,0,sizeof(pkt->rs_sizes[0])*(KEYAK_GPU_BUF_SLOTS));
    memset(pkt->ra_sizes,0,sizeof(pkt->ra_sizes[0])*(KEYAK_GPU_BUF_SLOTS));

    if (pkt->input_bytes_copied < pkt->input_size)
    {
        for (i=0; i < KEYAK_GPU_BUF_SLOTS; i++)
        {
            uint32_t tocopy = MIN(PISTON_RS * KEYAK_NUM_PISTONS, pkt->input_size - pkt->input_bytes_copied);
            memmove(pkt->merged + j, pkt->input + pkt->input_bytes_copied, tocopy);
            pkt->rs_sizes[l++] = tocopy;
            j += (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS);
            pkt->input_bytes_copied += tocopy;
            if (pkt->input_bytes_copied == pkt->input_size)
            {
                i++;
                break;
            }
        }
    }
    total_blocks = i;

    if (pkt->metadata_bytes_copied < pkt->metadata_size)
    {
        j=0,l=0;
        for (i=0; i < KEYAK_GPU_BUF_SLOTS; i++)
        {
            uint32_t tocopy = MIN(PISTON_RA * KEYAK_NUM_PISTONS, pkt->metadata_size - pkt->metadata_bytes_copied);
            memmove(pkt->merged + j + PISTON_RS * KEYAK_NUM_PISTONS, pkt->metadata + pkt->metadata_bytes_copied, tocopy);
            pkt->ra_sizes[l++] = tocopy;
            j += (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS);
            pkt->metadata_bytes_copied += tocopy;
            if (pkt->metadata_bytes_copied == pkt->metadata_size)
            {
                i++;
                break;
            }
        }
    }
    if (i > total_blocks)
    {
        total_blocks = i;
    }
    // sanity check
    assert(total_blocks <= KEYAK_GPU_BUF_SLOTS);

    uint8_t * ptr;

    ptr = e->p_coalesced;
    HANDLE_ERROR(cudaMemcpyAsync( ptr, pkt->merged,
                total_blocks * KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS, cudaMemcpyHostToDevice, e->stream));

    return ptr;
}


void engine_init(Engine * e)
{
    memset(e,0,sizeof(Engine));
    e->phase = EngineFresh;

    // TODO consider making these all one contiguous block or even different memories
    HANDLE_ERROR(cudaStreamCreate(&e->stream));
    HANDLE_ERROR(cudaMalloc(&e->p_in, PISTON_RS * KEYAK_NUM_PISTONS ));

    HANDLE_ERROR(cudaMalloc(&e->p_out, PISTON_RS * KEYAK_NUM_PISTONS * KEYAK_GPU_BUF_SLOTS ));
    HANDLE_ERROR(cudaMalloc(&e->p_state, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_tmp, KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_tag, KEYAK_TAG_SIZE/8));
    
    //e->p_coalesced = ENGINE_INPUT;
    HANDLE_ERROR(cudaMalloc(&e->p_coalesced, KEYAK_NUM_PISTONS * KEYAK_STATE_SIZE * KEYAK_GPU_BUF_SLOTS));

    HANDLE_ERROR(cudaMemset(e->p_state,0, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMemset(e->p_tmp,0,KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS ));

    // for lazy allocator
    HANDLE_ERROR(cudaMemset(e->p_in,0,PISTON_RS * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMemset(e->p_out,0,PISTON_RS * KEYAK_NUM_PISTONS * KEYAK_GPU_BUF_SLOTS ));
    HANDLE_ERROR(cudaMemset(e->p_tag,0, KEYAK_TAG_SIZE/8));
    HANDLE_ERROR(cudaMemset(e->p_coalesced,0,KEYAK_NUM_PISTONS * KEYAK_STATE_SIZE * KEYAK_GPU_BUF_SLOTS));


    HANDLE_ERROR(cudaGetSymbolAddress((void**)&e->p_offsets_cprime, OFFSETS_CPRIME));
    HANDLE_ERROR(cudaGetSymbolAddress((void**)&e->p_offsets_zero, OFFSETS_ZERO));
    HANDLE_ERROR(cudaGetSymbolAddress((void**)&e->p_offsets_1tag, OFFSETS_1TAG));

    gpu_init_keccak_tables();
}

void engine_destroy(Engine * e)
{
    engine_sync();
    cudaStreamDestroy(e->stream);
    cudaFree(e->p_in);
    cudaFree(e->p_out);
    cudaFree(e->p_tmp);
    cudaFree(e->p_state);
    cudaFree(e->p_coalesced);
}

void engine_restart(Engine * e)
{
    e->phase = EngineFresh;

    HANDLE_ERROR(cudaMemsetAsync(e->p_state,0, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS, e->stream ));
}

// offsets is GPU owned
void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets, uint8_t * dst, uint8_t size)
{
    piston_spark<<<KEYAK_NUM_PISTONS,PERMUTE_THREADS, 0, e->stream>>>
        (e->p_state, eom, offsets,dst,size);
}

// buf is GPU owned
void engine_get_tags_gpu(Engine * e, uint8_t * buf, uint8_t * L)
{
    assert(e->phase == EngineEndOfMessage);
    engine_spark(e, 1,e->p_offsets_cprime, buf, KEYAK_CPRIME/8);
    e->phase = EngineFresh;
}


void engine_get_tags_super(Engine * e, uint8_t * T, uint8_t * L)
{
    assert(e->phase == EngineEndOfMessage);
    e->phase = EngineFresh;
    HANDLE_ERROR(
            cudaMemcpyAsync(T,
                e->p_tag,
                (KEYAK_TAG_SIZE / 8), cudaMemcpyDeviceToHost, e->stream)
            );

}


void engine_get_tags(Engine * e, uint8_t * T, uint8_t * L)
{
    assert(e->phase == EngineEndOfMessage);
    uint8_t amt = (L == e->p_offsets_cprime) ? KEYAK_CPRIME/8 : 0;
    engine_spark(e, 1, L, e->p_tmp, amt);

    // stage it so there is only one copy if possible
    if (L == e->p_offsets_cprime || L == e->p_offsets_zero)
    {
        if (L == e->p_offsets_cprime)
        {
            HANDLE_ERROR(
                    cudaMemcpyAsync(T,
                        e->p_tmp,
                        (KEYAK_CPRIME / 8) * KEYAK_NUM_PISTONS, cudaMemcpyDeviceToHost, e->stream)
                    );
        }
    }
    else
    {
        assert(KEYAK_TAG_SIZE/8 <= PISTON_RS);
        HANDLE_ERROR(
                cudaMemcpyAsync(T,
                    e->p_state,
                    KEYAK_TAG_SIZE/8, cudaMemcpyDeviceToHost, e->stream)
                );

    }
    e->phase = EngineFresh;
}

uint8_t offsets_zero[KEYAK_NUM_PISTONS];
void engine_precompute()
{
    memset(offsets_zero, 0, sizeof(offsets_zero));
}

void engine_inject_collective(Engine * e, uint8_t * X, uint32_t size, uint8_t dFlag, uint8_t fromHost)
{
    assert(e->phase == EngineFresh);
    uint8_t * ptr = X;

    if (dFlag)
    {
        X[size] = KEYAK_NUM_PISTONS;
        X[size+1] = 0;
        size += 2;
    }

    // TODO should support variable length
    assert(size < KEYAK_BUFFER_SIZE);


    // copy collective to gpu
    if (fromHost)
    {
        HANDLE_ERROR(cudaMemcpyAsync(e->p_tmp,X,
                    size,
                    cudaMemcpyHostToDevice, e->stream));
        ptr = e->p_tmp;
    }

    uint32_t i;
    for (i=0; i < size; i += PISTON_RA)
    {
        if ( i + PISTON_RA >= size)
        {
            piston_inject_uniform<<<KEYAK_NUM_PISTONS, PISTON_RA, 0, e->stream>>>(e->p_state,
                    ptr, i, size - i, dFlag,0);
        }
        else
        {
            piston_inject_uniform<<<KEYAK_NUM_PISTONS, PISTON_RA, 0, e->stream>>>(e->p_state,
                    ptr, i, PISTON_RA, 0,1);

        }

    }

    e->phase = EngineEndOfMessage;

}

void engine_inject(Engine * e, uint8_t * A, uint8_t doSpark, uint32_t amt)
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
        piston_inject_seq<<<KEYAK_NUM_PISTONS, PISTON_RA, 0, e->stream>>>
        (e->p_state, A, amt, cryptingFlag, doSpark);
    }

    if (doSpark)
    {
        e->phase = EngineFresh;
    }
    else
    {
        e->phase = EngineEndOfMessage;
    }
}

// I is a GPU owned buffer
// O is a GPU owned buffer
void engine_crypt(Engine * e, uint8_t * I, uint8_t * O, uint8_t unwrapFlag, uint32_t rs_amt,
            uint8_t * A, uint8_t doSpark, uint32_t ra_size, uint8_t cryptingFlag,
            uint32_t input_size, uint32_t input_bytes_processed, uint32_t metadata_size, uint32_t metadata_bytes_processed)
{

    assert(e->phase == EngineFresh);

    piston_crypt_super<<<KEYAK_NUM_PISTONS,KEYAK_STATE_SIZE, 0, e->stream>>>
        (I,O,e->p_state,rs_amt, unwrapFlag, A,rs_amt,cryptingFlag, doSpark,
            input_size, input_bytes_processed, metadata_size, metadata_bytes_processed,
            e->p_tag, e->p_tmp
         );

    if (doSpark)
    {
        e->phase = EngineFresh;
    }
    else
    {
        e->phase = EngineEndOfMessage;
    }

}

void engine_yield(Engine * e, uint8_t * buf, size_t size)
{
    HANDLE_ERROR(cudaMemcpyAsync(buf, e->p_out,
                size,
                cudaMemcpyDeviceToHost, e->stream));
}

void engine_sync()
{
    HANDLE_ERROR(cudaDeviceSynchronize());
}

