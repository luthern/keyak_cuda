#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include "motorist.h"
#include "engine.h"
#include "piston.h"
#include "misc.h"
#include "defs.h"

void motorist_init(Motorist * m)
{
    uint8_t i;

    m->phase = MotoristReady;

    for (i = 0; i < KEYAK_NUM_PISTONS; i++)
    {
        piston_init(&m->pistons[i]);
    }
    engine_init(&m->engine, m->pistons);
}

void motorist_restart(Motorist * m)
{
    uint8_t i;
    m->phase = MotoristReady;

    for(i=0; i < KEYAK_NUM_PISTONS; i++)
    {
        piston_restart(&m->pistons[i]);
    }

    engine_restart(&m->engine);
}

static void make_knot(Motorist * m)
{
    Buffer Tprime;
    int i = KEYAK_NUM_PISTONS;
    uint8_t primes[KEYAK_NUM_PISTONS];
    buffer_init(&Tprime, NULL, 0);
    while(i--)
    {
        primes[i] = KEYAK_CPRIME/8;
    }

    engine_get_tags(&m->engine, &Tprime, primes);

    buffer_seek(&Tprime, 0);

    engine_inject_collective(&m->engine, &Tprime, 0);
}

void motorist_setup()
{

}

// 1 success
// 0 fail
static int handle_tag(Motorist * m, uint8_t tagFlag, Buffer * T,
                    uint8_t unwrapFlag)
{
    Buffer Tprime;
    buffer_init(&Tprime, NULL, 0);
    uint8_t offsets[KEYAK_NUM_PISTONS];
    memset(offsets, 0, sizeof(offsets));

    if (!tagFlag)
    {
        engine_get_tags(&m->engine,&Tprime, offsets);
    }
    else
    {
        offsets[0] = KEYAK_TAG_SIZE / 8;
        engine_get_tags(&m->engine,&Tprime, offsets);
        if (!unwrapFlag)
        {
            buffer_clone(T,&Tprime);
        }
        else if (!buffer_same(&Tprime,T))
        {
            m->phase = MotoristFailed;
            return 0;
        }
    }
    return 1;
}

struct timer tinject;
struct timer tcrypt;
struct timer tknot;
struct timer ttag;

void motorist_timers_end()
{
    timer_end(&tinject);
    timer_end(&tcrypt);
    timer_end(&tknot);
    timer_end(&ttag);
}

void motorist_wrap(Motorist * m, Buffer * I, Buffer * O, Buffer * A,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag)
{
    assert(m->phase == MotoristRiding);
    if (!buffer_has_more(I) && !buffer_has_more(A))
    {
        timer_start(&tinject, "engine_inject");
        engine_inject(&m->engine,A);
        timer_accum(&tinject);
    }

    while(buffer_has_more(I))
    {
        timer_start(&tcrypt, "engine_crypt");
        engine_crypt(&m->engine, I, O, unwrapFlag);
        timer_accum(&tcrypt);

        timer_start(&tinject, "engine_inject");
        engine_inject(&m->engine,A);
        timer_accum(&tinject);
    }

    while(buffer_has_more(A))
    {
        timer_start(&tinject, "engine_inject");
        engine_inject(&m->engine,A);
        timer_accum(&tinject);
    }

    if (KEYAK_NUM_PISTONS > 1 || forgetFlag)
    {
        timer_start(&tknot, "make_knot");
        printf("make_knot\n");
        make_knot(m);
        timer_accum(&tknot);
    }
    timer_start(&ttag, "handle_tag");
    int r = handle_tag(m, 1, T, unwrapFlag);
    timer_accum(&ttag);
}

uint8_t motorist_start_engine(Motorist * m, Buffer * suv, uint8_t tagFlag,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag)
{
    assert(m->phase == MotoristReady);

    engine_inject_collective(&m->engine, suv, 1);
    
    printf("Rs: %d\n", PISTON_RS);
    printf("Ra: %d\n", PISTON_RA);

    if (forgetFlag)
    {
        make_knot(m);
    }

    uint8_t r = handle_tag(m, tagFlag, T, unwrapFlag);

    if (r)
    {
        m->phase = MotoristRiding;
    }
    return r;
}


