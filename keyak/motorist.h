#ifndef _MOTORIST_H_
#define _MOTORIST_H_

#include "piston.h"
#include "engine.h"
#include "defs.h"

typedef enum
{
    MotoristReady = 0,
    MotoristRiding,
    MotoristFailed,
} MotoristState;

typedef struct _Motorist
{
    MotoristState phase;
    Engine engine;

#define GPU_NUM_INPUTS  8
    // gpu
    // TODO make buf word size for performance
    uint8_t input_buf[KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * GPU_NUM_INPUTS];
    uint32_t input_rs_size[GPU_NUM_INPUTS];
    uint32_t input_ra_size[GPU_NUM_INPUTS];
} Motorist;

void motorist_init(Motorist * m);

void motorist_restart(Motorist * m);

uint8_t motorist_start_engine(Motorist * m, Buffer * suv, uint8_t tagFlag,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag);

void motorist_wrap(Motorist * m, Packet * pkt, Buffer * O,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag);


#endif 
