
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "keyak.h"
#include "motorist.h"
#include "misc.h"


void keyak_init(Keyak* k, uint32_t b, uint32_t nr, uint32_t c, uint32_t t)
{
    k->W = MAX(b/25,8);
    k->c = c;
    motorist_init(&k->motorist, k->W, c, t);
    buffer_init(&k->T,NULL,0);
    buffer_init(&k->SUV,NULL,0);
}

void keyak_restart(Keyak * k)
{
    motorist_restart(&k->motorist);
    k->T.offset = 0;
    k->SUV.offset = 0;
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


void keyak_encrypt(Keyak * k, uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen)
{
    buffer_init(&k->I,data, datalen);
    buffer_init(&k->O,NULL, 0);
    buffer_init(&k->A,metadata, metalen);

    motorist_start_engine(&k->motorist, &k->SUV, 0, &k->T, 0, 0);

    motorist_wrap(&k->motorist,&k->I,&k->O,&k->A, &k->T, 0, 0);
}

void keyak_decrypt(Keyak * k, uint8_t * data, uint32_t datalen, 
                    uint8_t * metadata, uint32_t metalen, 
                    uint8_t * tag, uint32_t taglen)
{
    Buffer tagbuf;
    motorist_start_engine(&k->motorist, &k->SUV, 0, &k->T, 0, 0);
    buffer_init(&k->I,data, datalen);
    buffer_init(&k->O,NULL, 0);
    buffer_init(&k->A,metadata, metalen);
    buffer_init(&tagbuf, tag, taglen);
    motorist_wrap(&k->motorist,&k->I,&k->O,&k->A, &tagbuf, 1, 0);

    if (k->motorist.phase == MotoristFailed)
    {
        fprintf(stderr,"authentication failed\n");
        exit(1);
    }

}




