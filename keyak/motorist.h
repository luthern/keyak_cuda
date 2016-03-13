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
    uint32_t W;
    uint8_t pi;
    uint32_t c;
    uint32_t t;

    uint32_t Rs;
    uint32_t Ra;

    uint32_t cprime;

    MotoristState phase;

    Engine engine;

    Piston pistons[KEYAK_NUM_PISTONS];
} Motorist;

void motorist_init(Motorist * m, uint32_t W,
                    uint32_t c, uint32_t t);

void motorist_restart(Motorist * m);

uint8_t motorist_start_engine(Motorist * m, Buffer * suv, uint8_t tagFlag,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag);

void motorist_wrap(Motorist * m, Buffer * I, Buffer * O, Buffer * A,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag);


#endif 
