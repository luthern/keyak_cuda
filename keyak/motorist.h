#ifndef _MOTORIST_H_
#define _MOTORIST_H_

#include "piston.h"
#include "engine.h"
#include "defs.h"

#define MOTORIST_NOT_WRAPPED   1
#define MOTORIST_WRAPPED       0

typedef enum
{
    MotoristReady = 0,
    MotoristRiding,
    MotoristFailed,
    MotoristWrapped,
    MotoristDone,
    MotoristWaiting
} MotoristState;

typedef struct _Motorist
{
    Packet pkt;
    MotoristState phase;
    Engine engine;
    uint8_t * output;
    // for wrapping
    uint8_t tag[KEYAK_TAG_SIZE/8];
    //for unwrapping
    //uint8_t * auth_tag;

#define GPU_NUM_INPUTS  8
    // gpu
    // TODO make buf word size for performance
    uint8_t input_buf[KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * GPU_NUM_INPUTS];
    uint32_t input_rs_size[GPU_NUM_INPUTS];
    uint32_t input_ra_size[GPU_NUM_INPUTS];
} Motorist;

void motorist_init(Motorist * m);
//void motorist_fuel(Motorist * m, uint8_t * input, size_t ilen, uint8_t * metadata, size_t mlen);
void motorist_fuel(Motorist * m, uint8_t * input, uint32_t ilen, uint8_t * metadata, uint32_t mlen, uint8_t * tag);

void motorist_restart(Motorist * m);

uint8_t motorist_start_engine(Motorist * m, Buffer * suv, uint8_t tagFlag,
                    uint8_t * T, uint8_t unwrapFlag, uint8_t forgetFlag);

int motorist_wrap(Motorist * m, uint8_t unwrapFlag);
void motorist_authenticate(Motorist * m, uint8_t * T, uint8_t forgetFlag, uint8_t unwrapFlag);

void motorist_destroy(Motorist * m);

#endif 
