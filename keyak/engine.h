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
    uint32_t Et [KEYAK_NUM_PISTONS];
    EngineState phase;
} Engine;


void engine_init(Engine * e, Piston * pistons);
void engine_spark(Engine * e, uint8_t eom, uint32_t * offsets);

#endif
