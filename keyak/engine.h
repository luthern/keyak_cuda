#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "piston.h"
#include "defs.h"
    
typedef enum
{
    EngineFresh = 0,
    EngineCrypted,
    EngineEndOfCrypt,
    EngineEndOfMessage,
} EngineState;

typedef struct _Engine
{
    uint8_t Et [KEYAK_NUM_PISTONS];
    EngineState phase;

    // cuda
    uint8_t * p_in;
    uint8_t * p_out;
    uint8_t * p_state;
    uint8_t * p_tmp;
    uint8_t * p_offsets;

    uint8_t * p_coalesced;

    uint8_t * p_offsets_zero;
    uint8_t * p_offsets_cprime;
    uint8_t * p_offsets_1tag;

    cudaStream_t p_streams[KEYAK_NUM_PISTONS];

} Engine;

/**** optimizations */
typedef struct _Packet
{
    uint8_t * input;
    uint8_t * metadata;
    size_t  input_size;
    size_t  input_offset;
    size_t  metadata_size;
    size_t  metadata_offset;
    uint8_t merged[KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * KEYAK_GPU_BUF_SLOTS];
    uint32_t rs_sizes[KEYAK_GPU_BUF_SLOTS];
    uint32_t ra_sizes[KEYAK_GPU_BUF_SLOTS];
} Packet;

uint8_t * coalesce_gpu(Engine * e, Packet * pkt);
void dump_hex_cuda(uint8_t * buf, uint32_t size);
void engine_get_tags_gpu(Engine * e, uint8_t * buf, uint8_t * L);
void engine_yield(Engine * e, uint8_t * buf, uint32_t size);
/****               */

void engine_init(Engine * e);
void engine_restart(Engine * e);
void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets, uint8_t * dst, uint8_t size);
void engine_get_tags(Engine * e, Buffer * T, uint8_t * L);
void engine_inject(Engine * e, uint8_t * A, uint8_t isLeftovers,uint32_t amt);
void engine_inject_collective(Engine * e, uint8_t * X, uint32_t size, uint8_t dFlag, uint8_t fromHost);

void engine_crypt(Engine * e, uint8_t * I, uint8_t * O, uint8_t unwrapFlag, uint32_t amt,
            uint8_t * A, uint8_t doSpark, uint32_t size, uint8_t cryptingFlag);

void engine_precompute();
void engine_destroy(Engine * e);

#endif
