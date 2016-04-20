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

    engine_init(&m->engine);
}

void motorist_restart(Motorist * m)
{
    uint8_t i;
    m->phase = MotoristReady;

    engine_restart(&m->engine);
}

static void make_knot(Motorist * m)
{
    // TODO maybe make a piston function that does this
    engine_get_tags_gpu(&m->engine, m->engine.p_tmp, m->engine.p_offsets_cprime);

    engine_inject_collective(&m->engine, m->engine.p_tmp, KEYAK_NUM_PISTONS * KEYAK_CPRIME / 8, 0, 0);
}

void motorist_setup()
{

}

void motorist_destroy(Motorist * m)
{
    engine_destroy(&m->engine);
}

// 1 success
// 0 fail
static int handle_tag(Motorist * m, uint8_t tagFlag, Buffer * T,
                    uint8_t unwrapFlag)
{
    Buffer Tprime;
    buffer_init(&Tprime, NULL, 0);

    if (!tagFlag)
    {
        // move engine state along ..
        engine_get_tags(&m->engine,&Tprime, m->engine.p_offsets_zero);
    }
    else
    {
        if (!unwrapFlag)
        {
            engine_get_tags(&m->engine, T, m->engine.p_offsets_1tag);
            return 1;
        }

        // TODO do this on GPU
        engine_get_tags(&m->engine, &Tprime, m->engine.p_offsets_1tag);
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

void motorist_wrap(Motorist * m, Packet * pkt, Buffer * O,
                    Buffer * T, uint8_t unwrapFlag, uint8_t forgetFlag)
{
    assert(m->phase == MotoristRiding);
    if ((pkt->input_offset >= pkt->input_size) && (pkt->metadata_offset >= pkt->metadata_size))
    {
        timer_start(&tinject, "engine_inject");
        engine_inject(&m->engine,NULL,0,0);
        timer_accum(&tinject);
    }
    uint8_t bufsel = 0;

    int isize;
    int asize;
    uint32_t offset, rs_offset = 0, ra_offset = 0, out_offset = 0;

    uint8_t * block;


    do
    {
        block = coalesce_gpu(&m->engine, pkt);
        uint8_t i = 0;
        out_offset = 0;

        /*printf("\nabsorbed %ld/%ld input\n", pkt->input_offset,pkt->input_size);*/

        while((i < KEYAK_GPU_BUF_SLOTS) && pkt->rs_sizes[i])
        {
            
            uint8_t do_spark = 0;
            /*printf("i: %d < %d ? %d\n",i, KEYAK_GPU_BUF_SLOTS,i<KEYAK_GPU_BUF_SLOTS);*/

            offset = (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * i);

            timer_start(&tcrypt, "engine_crypt");

            // uint8_t * A, uint8_t doSpark, uint32_t size, uint8_t cryptingFlag)
            /*printf("crypting %d bytes\n",pkt->rs_sizes[i]);*/
            do_spark = ((rs_offset + pkt->rs_sizes[i]) < pkt->input_size || (ra_offset + pkt->ra_sizes[i]) < pkt->metadata_size);

            engine_crypt(&m->engine, block + offset, m->engine.p_out + out_offset, unwrapFlag, pkt->rs_sizes[i],
                    block + offset + PISTON_RS * KEYAK_NUM_PISTONS, do_spark, pkt->ra_sizes[i], 1,
                    pkt->input_size,pkt->metadata_size);

            out_offset += pkt->rs_sizes[i];

            rs_offset += pkt->rs_sizes[i];


            timer_accum(&tcrypt);


            ra_offset += pkt->ra_sizes[i];

            i++;
            break;
        }

        {
            out_offset = pkt->input_size;

        }

        timer_start(&tcrypt, "engine_crypt");
        engine_yield(&m->engine, O->buf, pkt->input_size);
        O->length += out_offset;
        timer_accum(&tcrypt);
    }
    while(pkt->input_offset < pkt->input_size);

    while(pkt->input_offset < pkt->input_size)
    {
        fprintf(stderr,"this should NOT happen!\n");
        block = coalesce_gpu(&m->engine, pkt);
        uint8_t i = 0;

        while(m->input_ra_size[i])
        {
            offset = (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * i);

            timer_start(&tinject, "engine_inject");

            engine_inject(&m->engine, block + offset + PISTON_RS * KEYAK_NUM_PISTONS,
                    (ra_offset + m->input_ra_size[i]) < pkt->metadata_size, m->input_ra_size[i]);

            ra_offset += m->input_ra_size[i];
            timer_accum(&tinject);
            i++;
        }
    }

    if (KEYAK_NUM_PISTONS > 1 || forgetFlag)
    {
        timer_start(&tknot, "make_knot");
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

    engine_inject_collective(&m->engine, suv->buf, suv->length, 1,1);
    
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



