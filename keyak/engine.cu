#include <string.h>
#include <assert.h>
#include "engine.h"
#include "misc.h"
#include "piston.h"

#include "utils.h"

// cuda does not support external linkage
#include "keccak.cu"
#include "piston.cu"

#if 1

void dump_state(Engine * e, int piston)
{
    uint8_t tmp[KEYAK_STATE_SIZE];

    HANDLE_ERROR(cudaMemcpy(tmp,e->p_state + piston * KEYAK_STATE_SIZE,
                            KEYAK_STATE_SIZE, cudaMemcpyDeviceToHost));
    dump_hex(tmp, sizeof(tmp));
}

#endif


void engine_init(Engine * e, Piston * pistons)
{
    memset(e,0,sizeof(Engine));
    e->pistons = pistons;
    e->phase = EngineFresh;

    // TODO consider making these all one contiguous block or even different memories
    HANDLE_ERROR(cudaMalloc(&e->p_in, PISTON_RS * KEYAK_NUM_PISTONS ));

    HANDLE_ERROR(cudaMalloc(&e->p_out, PISTON_RS * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_state, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_tmp, KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMalloc(&e->p_offsets, KEYAK_NUM_PISTONS ));

    HANDLE_ERROR(cudaMemset(e->p_state,0, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMemset(e->p_offsets,0,KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMemset(e->p_tmp,0,KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS ));
}

void engine_destroy(Engine * e)
{
    printf("engine_destroyed\n");
    cudaFree(e->p_in);
    cudaFree(e->p_out);
}

void engine_restart(Engine * e)
{
    e->phase = EngineFresh;
}

void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets)
{
    //printf("ENGINE_SPARK\n");

    // TODO pass offsets array
    cudaMemcpy(e->p_offsets, offsets, KEYAK_NUM_PISTONS, cudaMemcpyHostToDevice);

    //printf("spark state 1 : \n");
    //int j;
    //for (j=0; j < KEYAK_NUM_PISTONS; j++)
    //{
    //    printf("piston %d\n", j);
    //    dump_state(e,j);
    ///}


    piston_spark<<<KEYAK_NUM_PISTONS,1>>>
        (e->p_state, eom, e->p_offsets);

/*    printf("spark state 2 : \n");*/
    /*for (j=0; j < KEYAK_NUM_PISTONS; j++)*/
    /*{*/
        /*printf("piston %d\n", j);*/
        /*dump_state(e,j);*/
    /*}*/


    memmove(e->Et, offsets, KEYAK_NUM_PISTONS);
}

void engine_get_tags(Engine * e, Buffer * T, uint8_t * L)
{
    assert(e->phase == EngineEndOfMessage);
    uint8_t i;
/*    printf("get tags state 1: \n");*/
    /*for (j=0; j < KEYAK_NUM_PISTONS; j++)*/
    /*{*/
        /*printf("piston %d\n", j);*/
        /*dump_state(e,j);*/
    /*}*/
    engine_spark(e, 1, L);

  /*  printf("get tags state 2: \n");*/
    /*for (j=0; j < KEYAK_NUM_PISTONS; j++)*/
    /*{*/
        /*printf("piston %d\n", j);*/
        /*dump_state(e,j);*/
    /*}*/


    for (i = 0; i < KEYAK_NUM_PISTONS; i++)
    {
        if (L[i])
        {
            // TODO consider making one copy or making this async
            assert(L[i] <= PISTON_RS);
            HANDLE_ERROR(
                    cudaMemcpy(T->buf + T->length,
                                e->p_state + i * KEYAK_STATE_SIZE,
                                L[i], cudaMemcpyDeviceToHost)
                    );
            /*printf("copied tag bytes %d:\n",i);*/
            /*dump_hex(T->buf + T->length, L[i]);*/

            T->length += L[i];
        }
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
    //printf("ENGINE_INJECT\n");
    assert(
            e->phase == EngineCrypted ||
            e->phase == EngineEndOfCrypt ||
            e->phase == EngineFresh
            );
    uint8_t cryptingFlag = (
            e->phase == EngineCrypted ||
            e->phase == EngineEndOfCrypt
            );

    uint32_t amt = MIN(PISTON_RA*KEYAK_NUM_PISTONS, A->length - A->offset);
    uint8_t i;

    HANDLE_ERROR(
            cudaMemcpy(e->p_tmp, A->buf + A->offset, amt, cudaMemcpyHostToDevice)
            );

    //printf("injecting %d bytes\n", amt);
    
    //printf("inject state 1 : \n");
    //int j;
    //for (j=0; j < KEYAK_NUM_PISTONS; j++)
   // {
    //    printf("piston %d\n", j);
    //    dump_state(e,j);
    //}


    piston_inject_seq<<<KEYAK_NUM_PISTONS, PISTON_RA>>>
        (e->p_state, e->p_tmp, 0, amt, cryptingFlag);
    A->offset += amt;

//    printf("inject state 2 : \n");
//    for (j=0; j < KEYAK_NUM_PISTONS; j++)
//    {
//        printf("piston %d\n", j);
//        dump_state(e,j);
//    }

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

#if 0
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

static void dump_hash(Engine * e, int piston)
{
    uint8_t tmp[KEYAK_STATE_SIZE];

    HANDLE_ERROR(cudaMemcpy(tmp,e->p_state + piston * KEYAK_STATE_SIZE,
                            KEYAK_STATE_SIZE, cudaMemcpyDeviceToHost));
    PERMUTE(tmp);
    dump_hex(tmp, sizeof(tmp));
}
#endif



void engine_inject_collective(Engine * e, Buffer * X, uint8_t dFlag)
{
    assert(e->phase == EngineFresh);
    //printf("ENGINE_INJECT_COLLECTIVE\n");

    /*printf("collectively injecting %d bytes\n", X->length);*/

    /*dump_hex(X->buf, X->length);*/

    /*printf("COLLECTIVE INPUT STATE :\n");*/
    /*int j;*/
    /*for (j=0; j < KEYAK_NUM_PISTONS; j++)*/
    /*{*/
        /*[>printf("piston %d\n", j);<]*/
        /*dump_state(e,j);*/
    /*}*/



    if (dFlag)
    {
        /*printf("diversivefying\n");*/
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
            /*printf("injecting %d bytes\n", X->length - i);*/
            piston_inject_uniform<<<KEYAK_NUM_PISTONS, PISTON_RA>>>(e->p_state,
                    e->p_tmp, i, X->length - i, 0);
        }
        else
        {
            /*printf("injecting PISTON_RA bytes\n");*/
            piston_inject_uniform<<<KEYAK_NUM_PISTONS, PISTON_RA>>>(e->p_state,
                    e->p_tmp, i, PISTON_RA, 0);
            // data dependency
            piston_spark<<<KEYAK_NUM_PISTONS,1>>>
                (e->p_state, 0, NULL);

        }

/*        printf("COLLECTIVE INJECT STATE :\n");*/
        /*int j;*/
        /*for (j=0; j < KEYAK_NUM_PISTONS; j++)*/
        /*{*/
            /*printf("piston %d\n", j);*/
            /*dump_state(e,j);*/
        /*}*/



        // test
    }




    e->phase = EngineEndOfMessage;
}

//static int iter =0 ;
void engine_crypt(Engine * e, Buffer * I, Buffer * O, uint8_t unwrapFlag)
{

    assert(e->phase == EngineFresh);
    uint32_t amt = MIN(PISTON_RS*KEYAK_NUM_PISTONS, I->length - I->offset);

    //printf("state: \n");
    //int j;
    //for (j=0; j < KEYAK_NUM_PISTONS; j++)
    //{
    //    printf("piston %d\n", j);
    //    dump_state(e,j);
    //}

    //printf("plain text %d (offset: %d):\n",iter, I->offset);
    //dump_hex(I->buf, amt);

    // TODO consider copying more than 1 block
    // Copy block of input to GPU
    HANDLE_ERROR(cudaMemcpy(e->p_in,I->buf + I->offset,
                amt,
                cudaMemcpyHostToDevice));
    
    piston_crypt<<<KEYAK_NUM_PISTONS,PISTON_RS>>>
        (e->p_in,e->p_out,e->p_state,amt, unwrapFlag);

    // Copy the output of pistons
    assert(O->length + amt < KEYAK_BUFFER_SIZE);
    HANDLE_ERROR(cudaMemcpy(O->buf + O->length, e->p_out,
                amt,
                cudaMemcpyDeviceToHost));
    //printf("cipher text %d:\n",iter++);
    //dump_hex(O->buf + O->length, amt);

    O->length += amt;
    I->offset += amt;

    e->phase = buffer_has_more(I) ? EngineCrypted : EngineEndOfCrypt;

}


