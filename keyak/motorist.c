#include <stdint.h>
#include <stdio.h>
#include <assert.h>
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
    m->cprime = W*((c+W-1)/W)


    for (i = 0; i < KEYAK_NUM_PISTONS; i++)
    {
        piston_init(&m->pistons[i],m->Rs, m->Ra);
    }
    engine_init(&m->engine, m->pistons);
}

static void make_knot(Motorist * m)
{
    Buffer Tprime;
    int i = KEYAK_NUM_PISTONS;
    uint32_t primes[KEYAK_NUM_PISTONS];
    while(i--)
    {
        primes[i] = m->cprime;
    }

    buffer_init(&Tprime);
    engine_get_tags(&m->engine, &Tprime, primes);

    buffer_seek(&Tprime, 0);

    engine_inject_collective(&m->engine, &Tprime, 0);
}

// 1 success
// 0 fail
static int handle_tag(Motorist * m, uint8_t tagFlag, Buffer * T,
                    uint8_t unwrapFlag)
{
    Buffer Tprime;
    uint32_t offsets[KEYAK_NUM_PISTONS];
    memset(offsets, 0, sizeof(primes));

    if (!tagFlag)
    {
        engine_get_tags(&m->engine,Tprime, offsets);
    }
    else
    {
        offsets[0] = m->t/8;
        engine_get_tags(&m->engine,Tprime, offsets);
        if (!unwrapFlag)
        {
            buffer_clone(T,Tprime);
        }
        else if (!buffer_same(Tprime,T))
        {
            m->phase = MotoristFailed;
            return 0;
        }
    }
    return 1;
}

void motorist_wrap(Motorist * m, Buffer * I, Buffer * O, Buffer * A,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag)
{
    assert(m->phase == MotoristRiding);
    if (!buffer_has_more(I) && !buffer_has_more(A))
    {
        engine_inject(m->engine,A);
    }

    while(buffer_has_more(I))
    {
        engine_crypt(m->engine, I, O, unwrapFlag);
        engine_inject(m->engine,A);
    }

    while(buffer_has_more(A))
    {
        engine_inject(m->engine,A);
    }

    if (KEYAK_NUM_PISTONS > 1 || forgetFlag)
    {
        make_knot(m);
    }

    int r = handle_tag(true, T, unwrapFlag);

}
