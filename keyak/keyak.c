
#include <stdio.h>
#include <string.h>
#include "keyak.h"
#include "motorist.h"
#include "misc.h"
#include <assert.h>


void keyak_init(Keyak* k, uint32_t b, uint32_t nr, uint32_t c, uint32_t t)
{
    k->W = MAX(b/25,8);
    k->c = c;

    motorist_init(&k->motorist, k->W, c, t);
    
    buffer_init(&k->T,NULL,0);
    buffer_init(&k->SUV,NULL,0);
}


void keyak_set_suv(Keyak * k, uint8_t * key, uint32_t klen)
{
    int i;
    uint32_t lk = k->W/8 * CEIL(k->c + 9, k->W);
    assert(klen <= lk - 2);
    buffer_put(&k->SUV, lk);

    for(i = 0; i < klen; i++)
    {
        buffer_put(&k->SUV, key[i]);
    }
    buffer_put(&k->SUV, 1);
    while(k->SUV.length < lk)
    {
        buffer_put(&k->SUV, 0);
    }
    assert(k->SUV.buf[0] == k->SUV.length);
}

void keyak_add_nonce(Keyak * k, uint8_t * nonce, uint32_t len)
{
    int i;
    for (i = 0; i < len; i++)
    {
        buffer_put(&k->SUV, nonce[i]);
    }
}


void keyak_encrypt(Keyak * k, uint8_t * outcipher, uint8_t * outtag,  
                    uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen)
{
    Buffer I,O,A;

    buffer_init(&I,data, datalen);
    buffer_init(&O,NULL, 0);
    buffer_init(&A,metadata, metalen);

    motorist_start_engine(&k->motorist, &k->SUV, 0, &k->T, 0, 0);

    motorist_wrap(&k->motorist,&I,&O,&A, &k->T, 0, 0);

    memmove(outcipher, O.buf, O.length);
    memmove(outtag, k->T.buf, k->T.length);
}






