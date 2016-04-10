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

void dump_hex_cuda(uint8_t * buf, uint32_t size)
{
    char tbuf[2000];
    assert( size <= sizeof(tbuf));
    HANDLE_ERROR(cudaMemcpy(tbuf, buf, size, cudaMemcpyDeviceToHost));
    int i;
    for(i = 0; i < size; i++)
    {
        printf("%02hhx", tbuf[i]);
    }
    printf("\n");

}

#endif

// merge 2 cpu buffers and make 1 copy to GPU
uint8_t * coalesce_gpu(Engine * e, uint8_t * buf1, size_t size1, uint8_t * buf2, size_t size2)
{
    assert( size1 + size2 <= sizeof(e->coal1));

    if (size1)
    {
        memmove(e->coal1, buf1, size1);
    }
    if (size2)
    {
        memmove(e->coal1 + size1, buf2, size2);
    }

    HANDLE_ERROR(cudaMemcpyAsync( e->coal1_gpu, e->coal1, size1 + size2, cudaMemcpyHostToDevice));
    return e->coal1_gpu;
}

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
    
    HANDLE_ERROR(cudaMalloc(&e->coal1_gpu, sizeof(e->coal1)));
    HANDLE_ERROR(cudaMalloc(&e->coal2_gpu, sizeof(e->coal2)));

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

    HANDLE_ERROR(cudaMemset(e->p_in, 0, PISTON_RS * KEYAK_NUM_PISTONS ));

    HANDLE_ERROR(cudaMemset(e->p_out, 0, PISTON_RS * KEYAK_NUM_PISTONS ));

    HANDLE_ERROR(cudaMemset(e->p_state,0, KEYAK_STATE_SIZE * KEYAK_NUM_PISTONS ));
    HANDLE_ERROR(cudaMemset(e->p_offsets,0,KEYAK_NUM_PISTONS ));
    //HANDLE_ERROR(cudaMemset(e->p_tmp,0,KEYAK_BUFFER_SIZE * KEYAK_NUM_PISTONS ));
}

void engine_spark(Engine * e, uint8_t eom, uint8_t * offsets)
{
    //printf("ENGINE_SPARK\n");

    cudaMemcpyAsync(e->p_offsets, offsets, KEYAK_NUM_PISTONS, cudaMemcpyHostToDevice);

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
                    cudaMemcpyAsync(T->buf + T->length,
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

void engine_inject(Engine * e, uint8_t * A, uint8_t isLeftovers,uint32_t amt)
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


    // TODO this should be done in an init somewhere
    /*HANDLE_ERROR(*/
            /*cudaMemcpyAsync(e->p_tmp, A->buf + A->offset, amt, cudaMemcpyHostToDevice)*/
            /*);*/

    //printf("injecting %d bytes\n", amt);
    
    //printf("inject state 1 : \n");
    //int j;
    //for (j=0; j < KEYAK_NUM_PISTONS; j++)
   // {
    //    printf("piston %d\n", j);
    //    dump_state(e,j);
    //}


    if (amt)
    {
        piston_inject_seq<<<KEYAK_NUM_PISTONS, PISTON_RA>>>
        (e->p_state, A, 0, amt, cryptingFlag);
    }

//    printf("inject state 2 : \n");
//    for (j=0; j < KEYAK_NUM_PISTONS; j++)
//    {
//        printf("piston %d\n", j);
//        dump_state(e,j);
//    }

    if (e->phase == EngineCrypted || isLeftovers)
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
    /*printf("ENGINE_INJECT_COLLECTIVE\n");*/

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
    HANDLE_ERROR(cudaMemcpyAsync(e->p_tmp,X->buf,
                X->length,
                cudaMemcpyHostToDevice));

    // Duplicate for each piston but only make gpu2gpu copies
    // Async because no data dependency for any of them
    uint8_t j;
    for (j=1; j < KEYAK_NUM_PISTONS; j++)
    {
        HANDLE_ERROR(cudaMemcpyAsync(e->p_tmp + KEYAK_BUFFER_SIZE * j,e->p_tmp,
                    X->length,
                    cudaMemcpyDeviceToDevice));
    }
    if (dFlag)
    {
        diversify_pistons<<<1,KEYAK_NUM_PISTONS>>>(e->p_tmp, X->length,dFlag);
    }


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

    if (dFlag)
    {
        X->length -= 2;
    }


    e->phase = EngineEndOfMessage;
}

// I is a GPU owned buffer
// O is a CPU owned buffer
void engine_crypt(Engine * e, uint8_t * I, Buffer * O, uint8_t unwrapFlag, uint32_t amt)
{

    assert(e->phase == EngineFresh);

    //uint32_t amt = MIN(PISTON_RS*KEYAK_NUM_PISTONS, I->length - I->offset);

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
    //HANDLE_ERROR(cudaMemcpyAsync(e->p_in,I->buf + I->offset,
    //            amt,
    //            cudaMemcpyHostToDevice));
    
    // TODO is RISTON_RS i.e. 1-1 the best ratio here?

    piston_crypt<<<KEYAK_NUM_PISTONS,PISTON_RS>>>
        (I,e->p_out,e->p_state,amt, unwrapFlag);

    // Copy the output of pistons
    assert(O->length + amt < KEYAK_BUFFER_SIZE);
    HANDLE_ERROR(cudaMemcpyAsync(O->buf + O->length, e->p_out,
                amt,
                cudaMemcpyDeviceToHost));
    //printf("cipher text %d:\n",iter++);
    //dump_hex(O->buf + O->length, amt);

    O->length += amt;

    // TODO this in motorist
    //e->phase = amt == PISTON_RS * KEYAK_NUM_PISTONS ? EngineCrypted : EngineEndOfCrypt;

}


