#include <stdint.h>
#include <stdio.h>
#include "motorist.h"
#include "engine.h"
#include "piston.h"
#include "misc.h"
#include "defs.h"

void motorist_init(Motorist * m, uint32_t W,
                    uint32_t c, uint32_t t)
{
    uint8_t i;
    m->W = W;
    m->c = c;
    m->t = t;

    m->Rs = W * ((KEYAK_F_WIDTH - MAX(32, c))/W) / 8;
    m->Ra = W * ((KEYAK_F_WIDTH - 32)/W) / 8;

    m->phase = MotoristReady;


    for (i = 0; i < KEYAK_NUM_PISTONS; i++)
    {
        piston_init(&m->pistons[i],m->Rs, m->Ra);
    }
    engine_init(&m->engine, m->pistons);


}
