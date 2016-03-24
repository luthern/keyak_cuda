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

    Piston pistons[KEYAK_NUM_PISTONS];
} Motorist;

void motorist_init(Motorist * m);

void motorist_restart(Motorist * m);

uint8_t motorist_start_engine(Motorist * m, Buffer * suv, uint8_t tagFlag,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag);

void motorist_wrap(Motorist * m, Buffer * I, Buffer * O, Buffer * A,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag);


#endif 
