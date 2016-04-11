#ifndef _ENGINE_H_
#define _ENGINE_H_

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
    Piston * pistons;
    uint8_t Et [KEYAK_NUM_PISTONS];
    EngineState phase;

    // cuda
    uint8_t * p_in;
    uint8_t * p_out;
    uint8_t * p_state;
    uint8_t * p_tmp;
    uint8_t * p_offsets;


    uint8_t coal1[KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * KEYAK_GPU_BUF_SLOTS];
    uint8_t coal2[KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * KEYAK_GPU_BUF_SLOTS];

    uint8_t * coal1_gpu;
    uint8_t * coal2_gpu;

    uint8_t coalsel;

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
uint8_t * to_gpu(Engine * e, uint8_t bufsel, uint8_t * buf1, size_t size1);
void dump_hex_cuda(uint8_t * buf, uint32_t size);

/****               */

void engine_init(Engine * e, Piston * pistons);
void engine_restart(Engine * e);
void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets);
void engine_get_tags(Engine * e, Buffer * T, uint8_t * L);
void engine_inject(Engine * e, uint8_t * A, uint8_t isLeftovers,uint32_t amt);
void engine_inject_collective(Engine * e, Buffer * X, uint8_t dFlag);

void engine_crypt(Engine * e, uint8_t * I, Buffer * O, uint8_t unwrapFlag, uint32_t amt);

void engine_precompute();

#endif
