#include <string.h>
#include <assert.h>
#include "engine.h"
#include "misc.h"
#include "piston.h"

#include "utils.h"

// cuda does not support external linkage
#include "piston.cu"

void engine_init(Engine * e, Piston * pistons)
{
    memset(e,0,sizeof(Engine));
    e->pistons = pistons;
    e->phase = EngineFresh;
}

void engine_restart(Engine * e)
{
    e->phase = EngineFresh;
}

void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets)
{
    uint8_t i;
    for (i=0; i < KEYAK_NUM_PISTONS; i++)
    {
        piston_spark(&e->pistons[i],eom, offsets[i]);
    }
    memmove(e->Et, offsets, sizeof(uint8_t)*KEYAK_NUM_PISTONS);
}

void engine_get_tags(Engine * e, Buffer * T, uint8_t * L)
{
    assert(e->phase == EngineEndOfMessage);
    engine_spark(e, 1, L);
    uint8_t i;
    for (i = 0; i < KEYAK_NUM_PISTONS; i++)
    {
        piston_get_tag(&e->pistons[i], T, L[i]);
    }
    e->phase = EngineFresh;
}

uint8_t offsets_zero[KEYAK_NUM_PISTONS];
void engine_precompute()
{
    memset(offsets_zero, 0, sizeof(offsets_zero));
}

void engine_inject(Engine * e, Buffer * A)
{    
    assert(
            e->phase == EngineCrypted ||
            e->phase == EngineEndOfCrypt ||
            e->phase == EngineFresh
            );
    uint8_t cryptingFlag = (
            e->phase == EngineCrypted ||
            e->phase == EngineEndOfCrypt
            );

    uint8_t i;
    for(i=0; i < KEYAK_NUM_PISTONS; i++)
    {
        piston_inject(&e->pistons[i],A,cryptingFlag);
    }
    if (e->phase == EngineCrypted || buffer_has_more(A))
    {
        engine_spark(e,0, offsets_zero);
        e->phase = EngineFresh;
    }
    else
    {
        e->phase = EngineEndOfMessage;
    }
}

void engine_inject_collective(Engine * e, Buffer * X, uint8_t dFlag)
{
    assert(e->phase == EngineFresh);
    Buffer Xt[KEYAK_NUM_PISTONS];
    uint8_t i;
    for (i=0; i< KEYAK_NUM_PISTONS; i++)
    {
        buffer_init(Xt+i, NULL, 0);
    }

    while(buffer_has_more(X))
    {
        uint8_t b = buffer_get(X); 
        for (i=0; i< KEYAK_NUM_PISTONS; i++)
        {
            buffer_put(&Xt[i],b);
        }
    }

    if (dFlag)
    {
        for (i=0; i< KEYAK_NUM_PISTONS; i++)
        {
            buffer_put(&Xt[i],KEYAK_NUM_PISTONS);
            buffer_put(&Xt[i],i);
        }
    }
    for (i=0; i< KEYAK_NUM_PISTONS; i++)
    {
        buffer_seek(&Xt[i],0);
    }

    while(buffer_has_more(Xt))
    {
        for (i=0; i< KEYAK_NUM_PISTONS; i++)
        {
            piston_inject(&e->pistons[i], &Xt[i], 0);
        }
        if (buffer_has_more(Xt))
        {
            uint8_t offsets[KEYAK_NUM_PISTONS];
            memset(offsets, 0, sizeof(offsets));
            engine_spark(e, 0, offsets);
        }
    }

    e->phase = EngineEndOfMessage;
}


static int _init_crypt = 0;
void engine_crypt(Engine * e, Buffer * I, Buffer * O, uint8_t unwrapFlag)
{
    assert(e->phase == EngineFresh);

    printf("start: %d  end: %d  leftover %d\n",
            I->offset, I->length, I->length - I->offset);
    printf("the total i can saturate is %d\n",
            PISTON_RS * KEYAK_NUM_PISTONS);

    uint8_t * dev_in;
    uint8_t * dev_out;
    uint8_t * dev_state;
    uint32_t amt = MIN(PISTON_RS*KEYAK_NUM_PISTONS, I->length - I->offset);

    HANDLE_ERROR(cudaMalloc(&dev_in, PISTON_RS * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&dev_out, PISTON_RS * KEYAK_NUM_PISTONS ));
    
    if (!_init_crypt)
    {
        HANDLE_ERROR(cudaMalloc(&dev_state, PISTON_RS * KEYAK_F_WIDTH / 8 ));
        HANDLE_ERROR(cudaMemset(dev_state,0,PISTON_RS * KEYAK_F_WIDTH / 8 ));
    }

    HANDLE_ERROR(cudaMemcpy(dev_in,I->buf + I->offset,
                amt,
                cudaMemcpyHostToDevice));


    // function call
    uint32_t numBlocks = amt / MAX_CUDA_THREADS_PER_BLOCK + 1;

    piston_crypt<<<numBlocks,MAX_CUDA_THREADS_PER_BLOCK>>>(dev_in,dev_out,dev_state,amt, unwrapFlag);

    debug();
    HANDLE_ERROR(cudaMemcpy(O->buf, dev_out,
                amt,
                cudaMemcpyDeviceToHost));

    printf("cipher text 1:\n");
    dump_hex(O->buf, amt);

    debug();

    I->offset += amt;

    //uint8_t i;
    //for (i=0; i < KEYAK_NUM_PISTONS; i++)
    //{
    //    printf("  piston %d\n", i);
    //    piston_crypt(&e->pistons[i], I, O, e->Et[i], unwrapFlag);
    //}

    e->phase = buffer_has_more(I) ? EngineCrypted : EngineEndOfCrypt;

    cudaFree(dev_in);
    cudaFree(dev_out);
    exit(1);
}


