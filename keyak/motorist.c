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

    memset(&m->pkt, 0, sizeof(Packet));

    engine_init(&m->engine);
}

void motorist_fuel(Motorist * m, uint8_t * input, uint32_t ilen, uint8_t * metadata, uint32_t mlen, uint8_t * tag)
{
    memset(&m->pkt, 0, sizeof(Packet));
    m->pkt.input = input;
    m->pkt.input_size = ilen;
    m->pkt.metadata = metadata;
    m->pkt.metadata_size = mlen;
    if (tag != NULL)
    {
        memmove(m->tag, tag, KEYAK_TAG_SIZE/8);
    }
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
static int handle_tag(Motorist * m, uint8_t tagFlag, uint8_t * T,
                    uint8_t unwrapFlag)
{
    uint8_t Tprime[KEYAK_TAG_SIZE/8];

    if (!tagFlag)
    {
        // move engine state along ..
        engine_get_tags(&m->engine,Tprime, m->engine.p_offsets_zero);
    }
    else
    {
        if (!unwrapFlag)
        {
            engine_get_tags_super(&m->engine, T, m->engine.p_offsets_1tag);
            
            return 1;
        }

        // TODO separate the get_tags from the check
        engine_get_tags_super(&m->engine, Tprime, m->engine.p_offsets_1tag);


        engine_sync();
        if (memcmp(Tprime, T, KEYAK_TAG_SIZE/8) != 0)
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
    /*timer_end(&tinject);*/
    /*timer_end(&tcrypt);*/
    /*timer_end(&tknot);*/
    /*timer_end(&ttag);*/
    /*timer_end(&starttag);*/
}

extern void dump_state(Engine * e, int piston);

static unsigned int total_amt(unsigned int * sizes)
{
    unsigned int amt = 0;
    int i = 0;
    for (; i < KEYAK_GPU_BUF_SLOTS; i++)
    {
        amt += sizes[i];
    }
    return amt;
}

int motorist_wrap(Motorist * m, uint8_t unwrapFlag)
{

    Packet * pkt = &m->pkt;

    uint32_t offset, out_offset = 0;

    uint8_t * block;

    if (m->time_to_copy)
    {
        engine_yield(&m->engine, m->output, m->copy_amt);
        m->time_to_copy = 0;
        m->copy_amt = 0;
        return MOTORIST_NOT_WRAPPED;
    }

    if (!m->movingMem)
    {
        block = coalesce_gpu(&m->engine, pkt);
        m->mem = block;
        m->movingMem = 1;
        return MOTORIST_NOT_WRAPPED;
    }
    else
    {
        block = m->mem;
        m->movingMem = 0;
        m->mem = NULL;
    }


    if ((pkt->input_bytes_copied < pkt->input_size) || block)
    {
            /*block = coalesce_gpu(&m->engine, pkt);*/
        uint8_t i = 0;
        out_offset = 0;

        int ra_amt = total_amt(pkt->ra_sizes);
        int rs_amt = total_amt(pkt->rs_sizes);

        engine_crypt(&m->engine, block, m->engine.p_out, unwrapFlag, rs_amt,
                block, 0l, ra_amt, 1,
                pkt->input_size, pkt->input_bytes_processed, pkt->metadata_size, pkt->metadata_bytes_processed);

        pkt->input_bytes_processed += rs_amt;
        pkt->metadata_bytes_processed += ra_amt;

        m->time_to_copy = 1;
        m->copy_amt = rs_amt;

        if (pkt->input_bytes_copied < pkt->input_size)
        {
            return MOTORIST_NOT_WRAPPED;
        }
    }

    if (pkt->metadata_bytes_copied < pkt->metadata_size)
    {
        printf("this should not happen\n");
        exit(1);
        block = coalesce_gpu(&m->engine, pkt);
        uint8_t i = 0;

        while(m->input_ra_size[i])
        {
            offset = (KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS * i);

            /*timer_start(&tinject, "engine_inject");*/

            engine_inject(&m->engine, block + offset + PISTON_RS * KEYAK_NUM_PISTONS,
                    (pkt->metadata_bytes_processed + m->input_ra_size[i]) < pkt->metadata_size, m->input_ra_size[i]);

            pkt->metadata_bytes_processed += m->input_ra_size[i];
            /*timer_accum(&tinject);*/
            i++;
        }
        if (pkt->metadata_bytes_copied < pkt->metadata_size)
        {
            return MOTORIST_NOT_WRAPPED;
        }
    }

    m->phase = MotoristWrapped;

    return MOTORIST_WRAPPED;
}

void motorist_authenticate(Motorist * m, uint8_t * T, uint8_t forgetFlag, uint8_t unwrapFlag)
{
    int r = handle_tag(m, 1, T, unwrapFlag);
    m->phase = MotoristDone;
}

uint8_t motorist_start_engine(Motorist * m, Buffer * suv, uint8_t tagFlag,
                    uint8_t * T, uint8_t unwrapFlag, uint8_t forgetFlag)
{
    assert(m->phase == MotoristReady);

    engine_inject_collective(&m->engine, suv->buf, suv->length, 1,1);

    m->key_injected = 0;

    if (forgetFlag)
    {
        make_knot(m);
    }

    uint8_t r = handle_tag(m, tagFlag, T, unwrapFlag);

    if (r)
    {
        m->phase = MotoristRiding;
    }
    /*timer_accum(&starttag);*/
    return r;
}


