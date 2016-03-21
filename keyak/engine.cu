#include <string.h>
#include <assert.h>
#include "engine.h"
#include "misc.h"
#include "piston.h"

#include "utils.h"

// cuda does not support external linkage
#include "keccak.cu"
#include "piston.cu"




void engine_init(Engine * e, Piston * pistons)
{
    memset(e,0,sizeof(Engine));
    e->pistons = pistons;
    e->phase = EngineFresh;

    // TODO consider making this one contiguous block
    HANDLE_ERROR(cudaMalloc(&e->p_in, PISTON_RS * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_out, PISTON_RS * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_state, PISTON_RS * KEYAK_F_WIDTH / 8 ));
    HANDLE_ERROR(cudaMalloc(&e->p_tmp, KEYAK_BUFFER_SIZE * 8 ));

    HANDLE_ERROR(cudaMemset(e->p_state,0,PISTON_RS * KEYAK_F_WIDTH / 8 ));
}

void engine_destroy(Engine * e)
{
    cudaFree(e->p_in);
    cudaFree(e->p_out);
}

void engine_restart(Engine * e)
{
    e->phase = EngineFresh;
}

void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets)
{
    piston_spark<<<KEYAK_NUM_PISTONS,1>>>
        (e->p_state, eom, offsets);
    memmove(e->Et, offsets, KEYAK_NUM_PISTONS);
   
    /*
    uint8_t i;
    for (i=0; i < KEYAK_NUM_PISTONS; i++)
    {
        piston_spark(&e->pistons[i],eom, offsets[i]);
    }
    memmove(e->Et, offsets, KEYAK_NUM_PISTONS);
    */
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
        // piston_inject(&e->pistons[i],A,cryptingFlag);
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

static void dump_tmp_buf(Engine * e)
{
    uint8_t tmp[KEYAK_BUFFER_SIZE*KEYAK_NUM_PISTONS];

    HANDLE_ERROR(cudaMemcpy(tmp, e->p_tmp, sizeof(tmp),
                cudaMemcpyDeviceToHost));

    int offset = 0;
    int i;
    for (i=0; i < KEYAK_NUM_PISTONS; i++)
    {
        dump_hex(tmp + offset * KEYAK_BUFFER_SIZE, 100);
        printf("\r\n");
    }
}

static void dump_state(Engine * e, int piston)
{
    uint8_t tmp[KEYAK_STATE_SIZE];

    HANDLE_ERROR(cudaMemcpy(tmp,e->p_state + piston * KEYAK_STATE_SIZE,
                            KEYAK_STATE_SIZE, cudaMemcpyDeviceToHost));
    dump_hex(tmp, sizeof(tmp));
}

void engine_inject_collective(Engine * e, Buffer * X, uint8_t dFlag)
{
    assert(e->phase == EngineFresh);

    if (dFlag)
    {
        buffer_put(X,KEYAK_NUM_PISTONS);
        buffer_put(X,0);
    }

    // TODO should support variable length
    assert(X->length < KEYAK_BUFFER_SIZE);

    // copy collective to gpu
    HANDLE_ERROR(cudaMemcpy(e->p_tmp,X->buf,
                X->length,
                cudaMemcpyHostToDevice));

    // TODO check if its just better to make 8 copies
    // but i think device to device copying would be speedier than
    // host to device cuz pci bus
    dup_for_pistons<<<1,KEYAK_NUM_PISTONS>>>(e->p_tmp, X->length,dFlag);

    uint32_t i;
    for (i=0; i < X->length; i += PISTON_RA)
    {
        if ( i + PISTON_RA >= X->length)
        {
            printf("injecting %d bytes\n", X->length - i);
            piston_inject<<<KEYAK_NUM_PISTONS, X->length - i>>>(e->p_state,
                    e->p_tmp, i, X->length - i, 0);
        }
        else
        {
            printf("injecting PISTON_RA bytes\n");
            piston_inject<<<KEYAK_NUM_PISTONS, PISTON_RA>>>(e->p_state,
                    e->p_tmp, i, PISTON_RA, 0);
            // data dependency
            // TODO
            // call spark
        }
        // test
        int j = 0;
        for (j=0; j < KEYAK_NUM_PISTONS; j++)
        {
            printf("piston %d state: \n", j);
            dump_state(e,j);
        }

    }

    e->phase = EngineEndOfMessage;
/*
    uint8_t i;
    // 1 this is done
    for (i=0; i< KEYAK_NUM_PISTONS; i++)
    {
        buffer_init(Xt+i, NULL, 0);
    }
    // 2 done
    while(buffer_has_more(X))
    {
        uint8_t b = buffer_get(X); 
        for (i=0; i< KEYAK_NUM_PISTONS; i++)
        {
            buffer_put(&Xt[i],b);
        }
    }

    // 3 diversify done
    if (dFlag)
    {
        for (i=0; i< KEYAK_NUM_PISTONS; i++)
        {
            buffer_put(&Xt[i],KEYAK_NUM_PISTONS);
            buffer_put(&Xt[i],i);
        }
    }

    // (no need)
    for (i=0; i< KEYAK_NUM_PISTONS; i++)
    {
        buffer_seek(&Xt[i],0);
    }

    // TODO
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
*/
}


void engine_crypt(Engine * e, Buffer * I, Buffer * O, uint8_t unwrapFlag)
{
    assert(e->phase == EngineFresh);

    printf("start: %d  end: %d  leftover %d\n",
            I->offset, I->length, I->length - I->offset);
    printf("the total i can saturate is %d\n",
            PISTON_RS * KEYAK_NUM_PISTONS);

    uint32_t amt = MIN(PISTON_RS*KEYAK_NUM_PISTONS, I->length - I->offset);

    printf("plain text 1:\n");
    dump_hex(I->buf, amt);

    // TODO consider copying more than 1 block
    // Copy block of input to GPU
    debug();  // conor look back here
    HANDLE_ERROR(cudaMemcpy(e->p_in,I->buf + I->offset,
                amt,
                cudaMemcpyHostToDevice));
    debug();

    piston_crypt<<<KEYAK_NUM_PISTONS,PISTON_RS>>>
        (e->p_in,e->p_out,e->p_state,amt, unwrapFlag);

    // Copy the output of pistons
    HANDLE_ERROR(cudaMemcpy(O->buf, e->p_out,
                amt,
                cudaMemcpyDeviceToHost));

    printf("cipher text 1:\n");
    dump_hex(O->buf, amt);

    I->offset += amt;

    e->phase = buffer_has_more(I) ? EngineCrypted : EngineEndOfCrypt;

    exit(1);
}


