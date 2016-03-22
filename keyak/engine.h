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

} Engine;


void engine_init(Engine * e, Piston * pistons);
void engine_restart(Engine * e);
void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets);
void engine_get_tags(Engine * e, Buffer * T, uint8_t * L);
void engine_inject(Engine * e, Buffer * A);
void engine_inject_collective(Engine * e, Buffer * X, uint8_t dFlag);
void engine_crypt(Engine * e, Buffer * I, Buffer * O, uint8_t unwrapFlag);
void engine_precompute();

#endif
