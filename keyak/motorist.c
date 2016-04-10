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

    //for(i=0; i < KEYAK_NUM_PISTONS; i++)
    //{
    //    piston_restart(&m->pistons[i]);
    //}

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
        // move engine state along ..
        engine_get_tags(&m->engine,&Tprime, offsets);
    }
    else
    {
        offsets[0] = KEYAK_TAG_SIZE / 8;
        if (!unwrapFlag)
        {
            engine_get_tags(&m->engine,T, offsets);
            return 1;
        }
        
        engine_get_tags(&m->engine,&Tprime, offsets);
        if (!buffer_same(&Tprime,T))
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
struct timer starttag;

void motorist_timers_end()
{
    timer_end(&tinject);
    timer_end(&tcrypt);
    timer_end(&tknot);
    timer_end(&ttag);
    timer_end(&starttag);
}

extern void dump_state(Engine * e, int piston);
void motorist_wrap(Motorist * m, Buffer * I, Buffer * O, Buffer * A,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag)
{
    assert(m->phase == MotoristRiding);
    if (!buffer_has_more(I) && !buffer_has_more(A))
    {
        timer_start(&tinject, "engine_inject");
        engine_inject(&m->engine,NULL,0,0);
        timer_accum(&tinject);
    }
    uint8_t bufsel = 0;


    int isize = MIN(PISTON_RS*KEYAK_NUM_PISTONS, I->length - I->offset);
    int asize = MIN(PISTON_RA*KEYAK_NUM_PISTONS, A->length - A->offset);

    uint8_t * block = coalesce_gpu(&m->engine, bufsel, I->buf + I->offset, isize, A->buf + A->offset, asize);

    // TODO "double buffer" this
    while(buffer_has_more(I))
    {
        timer_start(&tcrypt, "engine_crypt");

        engine_crypt(&m->engine, block, O, unwrapFlag, isize);

        I->offset += isize;
        m->engine.phase = I->offset < I->length ? EngineCrypted : EngineEndOfCrypt;

        timer_accum(&tcrypt);

        timer_start(&tinject, "engine_inject");

        engine_inject(&m->engine,block + isize, (A->offset + asize) < A->length,asize);
        A->offset += asize;

        timer_accum(&tinject);

        if (buffer_has_more(I))
        {
            bufsel++;
            isize = MIN(PISTON_RS*KEYAK_NUM_PISTONS, I->length - I->offset);
            asize = MIN(PISTON_RA*KEYAK_NUM_PISTONS, A->length - A->offset);
            block = coalesce_gpu(&m->engine, bufsel, I->buf + I->offset, isize, A->buf + A->offset, asize);
        }

        /*printf("CRYPT STATE %d:\n", iter++);*/
        /*int j;*/
        /*for (j=0; j < KEYAK_NUM_PISTONS; j++)*/
        /*{*/
            /*printf("piston %d\n", j);*/
            /*dump_state(&m->engine,j);*/
        /*}*/
    }

    while(buffer_has_more(A))
    {
        A->offset += asize;
        asize = MIN(PISTON_RA*KEYAK_NUM_PISTONS, A->length - A->offset);
        bufsel++;
        block = coalesce_gpu(&m->engine, bufsel,NULL, 0, A->buf + A->offset, asize);

        timer_start(&tinject, "engine_inject");
        /*printf("theres more A\n");*/
        engine_inject(&m->engine, block, (A->offset + asize) < A->length, asize);
        timer_accum(&tinject);
    }

    if (KEYAK_NUM_PISTONS > 1 || forgetFlag)
    {
        timer_start(&tknot, "make_knot");
        /*printf("make_knot\n");*/
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
    timer_start(&starttag,"start_engine");
    assert(m->phase == MotoristReady);

    engine_inject_collective(&m->engine, suv, 1);
    
    if (forgetFlag)
    {
        make_knot(m);
    }

    uint8_t r = handle_tag(m, tagFlag, T, unwrapFlag);

    if (r)
    {
        m->phase = MotoristRiding;
    }
    timer_accum(&starttag);
    return r;
}


